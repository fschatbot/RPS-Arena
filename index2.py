# index.py
# ATTEMPTING HIGH PRIORITY MANUVER
import psutil, os
p = psutil.Process(os.getpid())
try:
	p.nice(psutil.REALTIME_PRIORITY_CLASS)
	print('Process priority upgraded to Realtime!!')
except Exception:
	# Non-windows or privilege error: ignore
	pass

from typing import Union
import random
from datetime import datetime
from math import pi, cos, sin
from tqdm import tqdm
import numpy as np
from scipy.spatial import cKDTree
from numba import njit
from PIL import Image, ImageDraw
import ffmpeg

# --- PyTorch RL dependencies ---
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# RL: Policy, Obs, Rewards
# -------------------------
class RLPolicyNet(nn.Module):
	def __init__(self, obs_dim=9, hidden=64):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_dim, hidden),
			nn.ReLU(),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Linear(hidden, 2)
		)

	def forward(self, x):
		out = self.net(x)
		return F.normalize(out, dim=-1)


class RLStrategy:
	"""
	Vectorized RL strategy:
	- compute_actions(positions_np, types_np) -> (N,2)
	- compute_and_store_rewards(...)
	- memory stores (index, obs_before, action_before, reward)
	"""

	def __init__(self, controlled_type: int, device: Union[str, torch.device] = "cpu"):
		self.controlled_type = int(controlled_type)
		self.device = torch.device(device)
		self.model = RLPolicyNet().to(self.device)
		self.memory = []

		# last-step buffers
		self._last_obs = None
		self._last_actions = None
		self._last_indices = None

	# -------------------------------------------------------------------------
	# OBSERVATION BUILDER (N,10)
	# -------------------------------------------------------------------------
	def _build_observations(self, positions: np.ndarray, types: np.ndarray):
		"""
		Observation = [
			onehot(3),
			dx_to_prey, dy_to_prey,
			dx_to_pred, dy_to_pred,
			dist_to_prey, dist_to_pred
		]
		"""
		pos = positions.astype(np.float32)
		types = types.astype(np.int64)
		N = pos.shape[0]

		# pairwise diffs (N,N,2)
		diff = pos[:, None, :] - pos[None, :, :]
		dist = np.linalg.norm(diff, axis=-1) + 1e-9
		np.fill_diagonal(dist, 1e9)

		# prey/pred types
		prey_of = (types + 1) % 3
		pred_of = (types + 2) % 3

		# vectorized masks
		prey_mask = (types[None, :] == prey_of[:, None])
		pred_mask = (types[None, :] == pred_of[:, None])

		# nearest prey & pred
		prey_dist = np.where(prey_mask, dist, 1e9).min(axis=1)
		pred_dist = np.where(pred_mask, dist, 1e9).min(axis=1)

		prey_idx = np.where(prey_mask, dist, 1e9).argmin(axis=1)
		pred_idx = np.where(pred_mask, dist, 1e9).argmin(axis=1)

		# direction vectors
		idxs = np.arange(N)

		prey_vec = diff[idxs, prey_idx]
		pred_vec = diff[idxs, pred_idx]

		# one-hot
		onehot = np.zeros((N, 3), dtype=np.float32)
		onehot[np.arange(N), types] = 1.0

		obs = np.concatenate([
			onehot,
			prey_vec.astype(np.float32),
			pred_vec.astype(np.float32),
			prey_dist.reshape(-1, 1).astype(np.float32),
			pred_dist.reshape(-1, 1).astype(np.float32)
		], axis=1)

		return obs  # (N,10)

	# -------------------------------------------------------------------------
	# ACTION COMPUTATION
	# -------------------------------------------------------------------------
	def compute_actions(self, positions: np.ndarray, types: np.ndarray):
		obs = self._build_observations(positions, types)
		obs_t = torch.from_numpy(obs).to(self.device)

		with torch.no_grad():
			actions_t = self.model(obs_t)        # (N,2)

		actions = actions_t.cpu().numpy().astype(np.float32)

		mask = (types == self.controlled_type)

		self._last_obs = obs[mask].copy()
		self._last_actions = actions[mask].copy()
		self._last_indices = np.nonzero(mask)[0].copy()

		return actions

	# -------------------------------------------------------------------------
	# REWARD COMPUTATION
	# -------------------------------------------------------------------------
	def compute_and_store_rewards(self, positions_after, types_after):
		if self._last_obs is None:
			return

		idxs = self._last_indices
		if len(idxs) == 0:
			self._last_obs = None
			self._last_actions = None
			self._last_indices = None
			return

		obs_after = self._build_observations(positions_after, types_after)
		obs_after_sel = obs_after[idxs]

		before_onehot = self._last_obs[:, :3]
		before_types = np.argmax(before_onehot, axis=1)
		after_types = types_after[idxs].astype(np.int64)

		converted = (after_types == self.controlled_type) & (before_types != self.controlled_type)
		eliminated = (after_types != self.controlled_type) & (before_types == self.controlled_type)

		dist_prey_after = obs_after_sel[:, 7].astype(np.float32)
		dist_pred_after = obs_after_sel[:, 8].astype(np.float32)

		moved = np.linalg.norm(self._last_actions, axis=1)
		moved_threshold = 1e-3

		rewards = np.zeros(len(idxs), dtype=np.float32)

		# base survival
		rewards += 0.01

		rewards += 1.0 * converted.astype(np.float32)
		rewards -= 1.0 * eliminated.astype(np.float32)

		# shaping
		rewards += 0.1 / (dist_prey_after + 1e-6)
		rewards -= 0.1 / (dist_pred_after + 1e-6)

		# movement penalty
		rewards -= (moved < moved_threshold).astype(np.float32) * 0.005 * (~converted)

		# store memory
		for j, global_index in enumerate(idxs):
			self.memory.append((
				int(global_index),
				self._last_obs[j].copy(),
				self._last_actions[j].copy(),
				float(rewards[j])
			))

		self._last_obs = None
		self._last_actions = None
		self._last_indices = None
	
	def train_step(self, batch_size=4096, gamma=0.99, lr=1e-3):
		"""
		Performs a single REINFORCE update using stored memory.
		Memory items: (idx, obs, action, reward).
		Vectorized batching.
		"""

		if len(self.memory) == 0:
			return 0.0  # nothing to train

		# Shuffle memory
		np.random.shuffle(self.memory)

		# Convert memory to arrays
		obs = np.array([m[1] for m in self.memory], dtype=np.float32)
		act = np.array([m[2] for m in self.memory], dtype=np.float32)
		rew = np.array([m[3] for m in self.memory], dtype=np.float32)

		# Compute discounted returns
		returns = np.zeros_like(rew, dtype=np.float32)
		running = 0.0
		for i in range(len(rew)-1, -1, -1):
			running = rew[i] + gamma * running
			returns[i] = running

		# Normalize returns (important for stability)
		returns = (returns - returns.mean()) / (returns.std() + 1e-6)

		# Move to torch
		obs_t = torch.from_numpy(obs).to(self.device)
		act_t = torch.from_numpy(act).to(self.device)
		ret_t = torch.from_numpy(returns).to(self.device)

		# Optimizer (created lazily)
		if not hasattr(self, "optimizer"):
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

		# Batch training
		total_loss = 0.0
		N = len(obs)

		for start in range(0, N, batch_size):
			end = start + batch_size

			b_obs = obs_t[start:end]
			b_act = act_t[start:end]
			b_ret = ret_t[start:end]

			# Forward pass
			pred = self.model(b_obs)                 # (B,2) normalized direction
			logp = -((pred - b_act)**2).sum(dim=-1)  # surrogate log-likelihood

			loss = -(logp * b_ret).mean()

			# Backprop
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			total_loss += loss.item()

		# Clear memory after update
		self.memory.clear()

		return total_loss

# -------------------------
# Game code (unchanged structure, integrated RL)
# -------------------------
class GameUniverse:
	def __init__(self, strategies):
		"""
		strategies: list length 3. Each entry either:
		  - a function: (current, surroundings, i) -> (ax, ay)
		  - an RLStrategy instance (controls that type)
		The list order corresponds to type 0,1,2 respectively.
		"""
		self.strategies = strategies
		self.radius = 10
		self.width = 800
		self.height = 800

		self.config = {
			'amount': 300,
			'clustering': 40,
			'dpi': 200,
			'stochasticity': 0.00,
			'win_edge': 0.5,
			'edge_behavior': 'bounce',
			'max_speed': 5,
		}
		self.objects = np.zeros((self.config['amount'], 5))  # [x,y,vx,vy,type]
		self.steps = 0

		# Sanity checks
		assert len(self.strategies) == 3, "There must be exactly three strategies"
		assert self.config['edge_behavior'] in ['wrap', 'bounce']
		assert self.config['amount'] > 0
		assert self.config['max_speed'] > 0
		assert self.config['stochasticity'] >= 0
		assert 0 <= self.config['win_edge'] <= 0.5
		assert self.width * self.height >= self.config['amount'] * (self.radius ** 2) * pi

		asset_paths = {0: "assets/rock.png", 1: "assets/paper.png", 2: "assets/scissor.png"}
		self.assets = {k: Image.open(v).convert("RGBA") for k, v in asset_paths.items()}
		self.assets = {k: v.resize((2 * self.radius, 2 * self.radius), Image.Resampling.LANCZOS) for k, v in self.assets.items()}

	def populate(self):
		assert self.config['amount'] % 3 == 0
		for i in range(self.config['amount']):
			if self.config['clustering'] > 0:
				cx, cy = self.width // 2, self.height // 2
				theta = (i % 3) * pi * 2 / 3 + pi / 2
				t_theta = min(self.width / (2 * abs(cos(theta)) + 1e-9),
							  self.height / (2 * abs(sin(theta)) + 1e-9)) / 2
				x = cx + t_theta * cos(theta) + random.gauss(0, self.config['clustering'])
				y = cy + t_theta * sin(theta) + random.gauss(0, self.config['clustering'])
			else:
				x = random.randint(0, self.width)
				y = random.randint(0, self.height)
			x = min(max(x, self.radius), self.width - self.radius)
			y = min(max(y, self.radius), self.height - self.radius)
			self.objects[i] = [x, y, 0, 0, i % 3]

	def step(self):
		assert self.objects.shape == (self.config['amount'], 5)
		N = self.objects.shape[0]

		# Precompute positions & types for RL strategies (full population)
		positions = self.objects[:, :2].astype(np.float32)
		types = self.objects[:, 4].astype(np.int64)

		# Prepare a container for updates (accelerations)
		updates = np.zeros((N, 2), dtype=np.float32)

		# If any strategy slots are RLStrategy, compute actions in batch per RLStrategy instance
		rl_actions_full = {}  # type_index -> (N,2) actions
		for t in range(3):
			strat = self.strategies[t]
			if isinstance(strat, RLStrategy):
				rl_actions_full[t] = strat.compute_actions(positions, types)

		# Compute updates per-entity:
		for i in range(N):
			t = int(self.objects[i, 4])
			strat = self.strategies[t]
			if isinstance(strat, RLStrategy):
				# Use precomputed batched action for this entity
				updates[i, :] = rl_actions_full[t][i]
			else:
				# Non-RL function: maintain compatibility with signature
				try:
					ax, ay = strat(self.objects[i], self.objects, i)
				except TypeError:
					# fallback if different signature
					ax, ay = strat(self.objects[i], self.objects)
				updates[i, 0] = float(ax)
				updates[i, 1] = float(ay)

		# Add stochasticity
		if self.config['stochasticity'] > 0:
			updates += np.random.uniform(-self.config['stochasticity'],
										 self.config['stochasticity'],
										 updates.shape).astype(np.float32)

		# normalize accelerations > 1
		mag = np.linalg.norm(updates, axis=1)
		over = mag > 1.0
		if np.any(over):
			updates[over] /= mag[over][:, np.newaxis]

		# Apply acceleration to velocity
		self.objects[:, 2:4] += updates
		mag_v = np.linalg.norm(self.objects[:, 2:4], axis=1)
		over_v = mag_v > self.config['max_speed']
		if np.any(over_v):
			self.objects[over_v, 2:4] /= mag_v[over_v][:, np.newaxis]

		# Apply velocity to position
		self.objects[:, :2] += self.objects[:, 2:4]

		# Edge behavior
		if self.config['edge_behavior'] == 'wrap':
			self.objects[:, 0] %= self.width
			self.objects[:, 1] %= self.height
		else:  # bounce
			self.objects[(self.objects[:, 0] < self.radius) |
						 (self.objects[:, 0] > self.width - self.radius), 2] *= -1
			self.objects[(self.objects[:, 1] < self.radius) |
						 (self.objects[:, 1] > self.height - self.radius), 3] *= -1
			self.objects[:, 0] = np.clip(self.objects[:, 0], self.radius + 1e-9, self.width - self.radius - 1e-9)
			self.objects[:, 1] = np.clip(self.objects[:, 1], self.radius + 1e-9, self.height - self.radius - 1e-9)

		# Handle collisions using cKDTree
		tree = cKDTree(self.objects[:, :2])
		collision_dist = 2.0 * self.radius
		pairs_set = tree.query_pairs(collision_dist)

		if len(pairs_set) > 0:
			pairs = np.array(list(pairs_set), dtype=np.intp)
			i_idx = pairs[:, 0]
			j_idx = pairs[:, 1]

			pos = self.objects[:, :2]
			vel = self.objects[:, 2:4]
			types_arr = self.objects[:, 4].astype(int)

			# Resolve elastic collisions
			for k in range(len(i_idx)):
				i = int(i_idx[k]); j = int(j_idx[k])
				dx = pos[j, 0] - pos[i, 0]; dy = pos[j, 1] - pos[i, 1]
				dist2 = dx * dx + dy * dy
				if dist2 == 0:
					dx = (np.random.random() - 0.5) * 1e-6
					dy = (np.random.random() - 0.5) * 1e-6
					dist2 = dx * dx + dy * dy
				dist = np.sqrt(dist2)
				nx = dx / (dist + 1e-12); ny = dy / (dist + 1e-12)
				vix = vel[i, 0]; viy = vel[i, 1]
				vjx = vel[j, 0]; vjy = vel[j, 1]
				rvx = vix - vjx; rvy = viy - vjy
				rel_vel_norm = rvx * nx + rvy * ny
				if rel_vel_norm < 0:
					vni = (vix * nx + viy * ny)
					vnj = (vjx * nx + vjy * ny)
					vti_x = vix - vni * nx; vti_y = viy - vni * ny
					vtj_x = vjx - vnj * nx; vtj_y = vjy - vnj * ny
					vni, vnj = vnj, vni
					vel[i, 0] = vti_x + vni * nx
					vel[i, 1] = vti_y + vni * ny
					vel[j, 0] = vtj_x + vnj * nx
					vel[j, 1] = vtj_y + vnj * ny
				penetration = 2.0 * self.radius - dist
				if penetration > 0:
					correction = 0.5 * penetration + 1e-9
					pos[i, 0] -= nx * correction
					pos[i, 1] -= ny * correction
					pos[j, 0] += nx * correction
					pos[j, 1] += ny * correction

			# Type resolution
			a = types_arr[i_idx]; b = types_arr[j_idx]
			win_chance = np.where((a + 2) % 3 == b, 0.5 + self.config['win_edge'], 0.5 - self.config['win_edge'])
			rolls = np.random.random(len(i_idx))
			a_wins = rolls < win_chance
			types_arr[j_idx[a_wins]] = a[a_wins]
			types_arr[i_idx[~a_wins]] = b[~a_wins]
			self.objects[:, 4] = types_arr

		# After all updates, allow RL strategies to compute rewards and store experiences
		positions_after = self.objects[:, :2].astype(np.float32)
		types_after = self.objects[:, 4].astype(np.int64)
		for t in range(3):
			strat = self.strategies[t]
			if isinstance(strat, RLStrategy):
				strat.compute_and_store_rewards(positions_after, types_after)

		self.steps += 1

	def is_victory(self) -> bool:
		unique_types = np.unique(self.objects[:, 4])
		return len(unique_types) == 1

	def draw(self):
		img = Image.new("RGB", (self.width, self.height), (255, 255, 255))		

		for obj in self.objects:
			x, y, _, _, obj_type = obj
			r = self.radius
			sprite = self.assets[int(obj_type)]
			# Rotate the sprite based on velocity direction
			vel_x, vel_y = obj[2], obj[3]
			angle = np.degrees(np.arctan2(vel_y, vel_x)) if (vel_x != 0) else 0.0
			rotated_sprite = sprite.rotate(-(angle + 90), resample=Image.Resampling.BICUBIC, expand=False)
			img.paste(rotated_sprite, (int(x - r), int(y - r)), rotated_sprite)
		return img

# --- existing strategies (chasing / weighted) ---
def chasing(current:np.ndarray, surroundings:np.ndarray, i):
	prey = surroundings[(surroundings[:, 4] == (current[4] + 1) % 3)]
	if len(prey) == 0:
		return 0.0, 0.0
	vec = prey[:, :2] - current[:2]
	mag = np.linalg.norm(vec, axis=1)
	best = np.argmin(mag)
	return (vec[best] / (mag[best] + 1e-9)).astype(np.float32)

@njit
def njit_weighted_chasing(current: np.ndarray, surroundings: np.ndarray, i: int):
	cur_type = int(current[4])
	N = surroundings.shape[0]
	weights = np.empty(N, dtype=np.float64)
	for j in range(N):
		t = int(surroundings[j, 4])
		d = (t - cur_type) % 3
		if d == 0:
			weights[j] = -0.5
		elif d == 1:
			weights[j] = -2.0
		else:
			weights[j] = 2.0
	diff = surroundings[:, :2] - current[:2]
	for j in range(N):
		dx = diff[j, 0]; dy = diff[j, 1]
		dist2 = dx * dx + dy * dy + 1e-9
		diff[j, 0] = dx / dist2
		diff[j, 1] = dy / dist2
	for j in range(N):
		w = weights[j]
		diff[j, 0] *= w; diff[j, 1] *= w
	move_x = 0.0; move_y = 0.0
	for j in range(N):
		move_x += diff[j, 0]; move_y += diff[j, 1]
	mag = (move_x * move_x + move_y * move_y) ** 0.5
	if mag > 0.0:
		return move_x / mag, move_y / mag
	else:
		return 0.0, 0.0

def weighted_chasing(current:np.ndarray, surroundings:np.ndarray, i:int):
	weight_map = {current[4]: 0, (current[4] + 1) % 3: -2, (current[4] + 2) % 3: 2}
	diff = surroundings[:, :2] - current[:2]
	distances = np.linalg.norm(diff, axis=1)**2 + 1e-9
	diff /= distances[:, np.newaxis]
	weights = np.array([weight_map[int(obj[4])] for obj in surroundings])
	diff *= weights[:, np.newaxis]
	move = np.sum(diff, axis=0)
	mag = np.linalg.norm(move)
	return (move / mag) if mag > 0 else (0, 0)

# -------------------------
# main
# -------------------------
def main():
	MAX_STEPS = 5_000

	# Create RLStrategy for type 0 (example). Place RLStrategy instance in the strategies list
	device = "cuda" if torch.cuda.is_available() else "cpu"
	rl_for_type0 = RLStrategy(controlled_type=0, device=device)

	# Example: type 0 -> RL, type 1 -> njit_weighted_chasing, type 2 -> njit_weighted_chasing
	universe = GameUniverse([rl_for_type0, njit_weighted_chasing, njit_weighted_chasing])
	universe.populate()

	output_name = f"exports/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
	process = (
		ffmpeg
		.input('pipe:', format='rawvideo', pix_fmt='rgb24',
			   s=f'{universe.width}x{universe.height}', framerate=24)
		.output(output_name, vcodec='libx264', pix_fmt='yuv420p', r=24, crf=18, loglevel='error')
		.overwrite_output()
		.run_async(pipe_stdin=True)
	)

	progress = tqdm(desc="Simulating", unit="step", total=MAX_STEPS)
	while not universe.is_victory():
		universe.step()
		progress.update(1)
		frame = universe.draw()
		process.stdin.write(np.asarray(frame).tobytes())
		if universe.steps % 50 == 0:
			# Periodic RL training step
			for strat in universe.strategies:
				if not isinstance(strat, RLStrategy): continue
				loss = strat.train_step()
				print(f"RL Strategy (type {strat.controlled_type}) training step completed. Loss: {loss:.6f}")
		
		if universe.steps >= MAX_STEPS:
			print("Reached maximum steps without victory. Exiting.")
			break

	progress.close()
	process.stdin.close()
	process.wait()
	print(f"Exported: {output_name}")

	# Optional: inspect RL memory sizes
	for idx, strat in enumerate(universe.strategies):
		if isinstance(strat, RLStrategy):
			print(f"RL strategy for type {strat.controlled_type} stored {len(strat.memory)} experience tuples.")

import cProfile
if __name__ == "__main__":
	cProfile.run('main()', 'profile_stats.prof')
