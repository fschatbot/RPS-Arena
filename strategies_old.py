import numpy as np
from numba import njit
from tqdm import tqdm

def chasing(current:np.ndarray, surroundings:np.ndarray, i):
	prey = surroundings[(surroundings[:, 4] == (current[4] - 1) % 3)]
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


import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from typing import List

Transition = namedtuple('Transition', ['obs', 'action', 'logprob', 'reward', 'done'])

class RLAgent(nn.Module):
	def __init__(self):
		super().__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# 1. Feature Extractor (Per Neighbor)
		# Input: (Batch, 5, 300) -> 5 features: dx, dy, vx, vy, type
		self.encoder = nn.Sequential(
			nn.Conv1d(5, 64, kernel_size=1), nn.ReLU(),
			nn.Conv1d(64, 128, kernel_size=1), nn.ReLU(),
		)

		# 2. Attention Mechanism
		# Learns "how important is this neighbor?"
		self.attn_layer = nn.Sequential(
			nn.Conv1d(128, 64, kernel_size=1), nn.Tanh(),
			nn.Conv1d(64, 1, kernel_size=1) # Output score per neighbor
		)

		# 3. Decision Head
		# Takes the weighted context vector and decides action
		self.head = nn.Sequential(
			nn.Linear(128, 64), nn.ReLU(),
			nn.Linear(64, 32), nn.ReLU(),
			nn.Linear(32, 2),
			nn.Tanh() # Output -1 to 1
		)
		self.to(self.device)

	def forward(self, x):
		# x: (Batch, 300, 5) -> Permute to (Batch, 5, 300)
		x = x.permute(0, 2, 1)
		
		# A. Extract Features for every neighbor: (B, 128, 300)
		features = self.encoder(x)
		
		# B. Compute Attention Scores: (B, 1, 300)
		attn_logits = self.attn_layer(features)
		# Softmax over the neighbors dimension (dim=2) to get probabilities
		attn_weights = torch.softmax(attn_logits, dim=2)
		
		# C. Weighted Sum (Attention Pooling)
		# Multiply features by weights and sum over neighbors
		# (B, 128, 300) * (B, 1, 300) -> Sum dim 2 -> (B, 128)
		context = torch.sum(features * attn_weights, dim=2)
		
		# D. Action
		action = self.head(context)
		return action


class RLStrategy:
	def __init__(self, controlled_type: int = 0, use_pretrained: bool = False):
		self.controlled_type = controlled_type
		self.use_pretrained = use_pretrained
		self.N = 300
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.policy = RLAgent()
		self.policy_old = RLAgent()
		self.policy_old.load_state_dict(self.policy.state_dict())
		self.policy_old.eval()

		self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

		# PPO Hyperparameters (perfect for this env)
		self.gamma = 0.99
		self.lam = 0.95
		self.clip_eps = 0.2
		self.entropy_coef = 0.01
		self.ppo_epochs = 4
		self.batch_size = 2048
		self.max_grad_norm = 0.5

		# Episode storage
		self.buffer_obs = []
		self.buffer_actions = []
		self.buffer_logprobs = []
		self.buffer_rewards = []
		self.buffer_dones = []

		self.recording_started = False
		self.prev_obs = None
		self.prev_board = None

	# ------------------------------------------------------------------ #
	# 1. Ego-centric observation from full board state
	# ------------------------------------------------------------------ #
	def _make_egocentric_obs(self, board: np.ndarray) -> torch.Tensor:
		pos = torch.from_numpy(board[:, :2]).float().to(self.device)
		vel = torch.from_numpy(board[:, 2:4]).float().to(self.device)
		typ = torch.from_numpy(board[:, 4]).long().to(self.device)

		# Relative positions
		dx = pos.unsqueeze(1) - pos.unsqueeze(0)  # (300, 300, 2)

		# Ego-centric type encoding
		my_type = typ.unsqueeze(1)
		other_type = typ.unsqueeze(0)
		diff = (other_type - my_type) % 3
		ego_type = torch.where(diff == 1, torch.tensor(1.0, device=self.device),
					 torch.where(diff == 2, torch.tensor(-1.0, device=self.device),
										   torch.tensor(0.0, device=self.device)))

		# Zero self
		idx = torch.arange(self.N, device=self.device)
		ego_type[idx, idx] = 0
		dx[idx, idx] = 0

		# Broadcast velocities
		vx = vel[:, 0:1].expand(-1, self.N)
		vy = vel[:, 1:2].expand(-1, self.N)

		obs = torch.stack([
			dx[..., 0], dx[..., 1],   # dx, dy
			vx, vy,                   # vx, vy
			ego_type.float()
		], dim=-1)  # (300, 300, 5)
		return obs
	
	def export_model(self, path: str):
		torch.save(self.policy.state_dict(), path)
	
	def import_model(self, path: str):
		self.policy.load_state_dict(torch.load(path, map_location=self.device))
		self.policy_old.load_state_dict(self.policy.state_dict())
		self.policy_old.eval()
		self.policy.to(self.device)

	# ------------------------------------------------------------------ #
	# 2. Action selection
	# ------------------------------------------------------------------ #
	def compute_actions(self, board_state: np.ndarray) -> np.ndarray:
		self.prev_board = board_state.copy()
		
		obs = self._make_egocentric_obs(board_state)
		self.prev_obs = obs
		my_mask = (board_state[:, 4] == self.controlled_type)

		if not my_mask.any():
			return np.zeros((self.N, 2), dtype=np.float32)

		my_obs = obs[my_mask]  # (M, 300, 5)

		with torch.no_grad():
			action = self.policy_old(my_obs).cpu().numpy()  # (M, 2)

		actions = np.zeros((self.N, 2), dtype=np.float32)
		actions[my_mask] = action
		return actions

	# ------------------------------------------------------------------ #
	# 3. Store transition + reward (training disabled if pretrained)
	# ------------------------------------------------------------------ #
	def compute_and_store_rewards(self, after_state: np.ndarray):
		if self.use_pretrained:
			return  # evaluation mode → no training

		assert self.prev_board is not None, "Previous board state is None"
		# assert np.all(self.prev_board == after_state), "Board state shape mismatch"

		reward = self._compute_reward(self.prev_board, after_state)
		# if not self.recording_started:
		# 	if np.isclose(reward, 0.0):
		# 		self.prev_board = after_state.copy()
		# 		return
		# 	else:
		# 		self.recording_started = True
		done = len(np.unique(after_state[:, 4])) == 1

		# Get observations and actions from previous step
		my_mask = (self.prev_board[:, 4] == self.controlled_type)

		if my_mask.any():
			my_prev_obs = self.prev_obs[my_mask]
			with torch.no_grad():
				mu = self.policy_old(my_prev_obs)
				dist = torch.distributions.Normal(mu, 0.1)
				action = dist.sample()
				logprob = dist.log_prob(action).sum(-1)
			
			count = len(action)

			self.buffer_obs.append(my_prev_obs.cpu())
			self.buffer_actions.append(action.cpu())
			self.buffer_logprobs.append(logprob.cpu())
			
			# For scalar rewards, we can just use extend
			self.buffer_rewards.extend([reward] * count)
			self.buffer_dones.extend([float(done)] * count)

		self.prev_board = after_state.copy()
		self.prev_obs = None

	# ------------------------------------------------------------------ #
	# 4. Asymmetric reward: predator death = jackpot
	# ------------------------------------------------------------------ #
	def _compute_reward(self, before: np.ndarray, after: np.ndarray) -> float:
		beforeT = before[:, 4]
		afterT = after[:, 4]
		my = self.controlled_type
		pred = (my + 1) % 3
		prey = (my + 2) % 3

		my_b = np.sum(beforeT == my)
		pred_b = np.sum(beforeT == pred)
		prey_b = np.sum(beforeT == prey)

		my_a = np.sum(afterT == my)
		pred_a = np.sum(afterT == pred)
		prey_a = np.sum(afterT == prey)

		pred_killed = pred_b - pred_a
		prey_killed = prey_b - prey_a
		my_died = my_b - my_a

		r = 5.0 * pred_killed + \
			1.5 * prey_killed - \
			8.0 * my_died

		# Terminal bonus
		if len(np.unique(after)) == 1:
			r += 20000.0 if after[0] == my else -15000.0

		# Next we add a small reward based on current population
		r += 2 * (my_a - pred_a - prey_a) / (my_a + pred_a + prey_a)

		# Next figure out the center of prey
		prey_positions = after[afterT == prey][:, :2]
		prey_center = np.mean(prey_positions, axis=0) if len(prey_positions) > 0 else np.array([0.0, 0.0])

		pred_positions = after[afterT == pred][:, :2]
		pred_center = np.mean(pred_positions, axis=0) if len(pred_positions) > 0 else np.array([0.0, 0.0])

		my_positions = after[afterT == my][:, :2]
		my_center = np.mean(my_positions, axis=0) if len(my_positions) > 0 else np.array([0.0, 0.0])

		r -= 0.01 * np.linalg.norm(my_center - prey_center)
		r += 0.01 * np.linalg.norm(my_center - pred_center)

		# # Dense shaping
		# total = len(after)
		# r += 25.0 * (my_a - my_b) / total
		# r -= 40.0 * pred_a / total

		return float(r)

	# ------------------------------------------------------------------ #
	# 5. Full PPO update after each episode
	# ------------------------------------------------------------------ #
	def _train_on_episode(self):
		if len(self.buffer_obs) == 0: return

		obs = torch.cat(self.buffer_obs).to(self.device)
		actions = torch.cat(self.buffer_actions).to(self.device)
		old_logprobs = torch.cat(self.buffer_logprobs).to(self.device)
		rewards = torch.tensor(self.buffer_rewards, device=self.device, dtype=torch.float32)
		dones = torch.tensor(self.buffer_dones, device=self.device, dtype=torch.float32)

		# Compute advantages using GAE
		with torch.no_grad():
			values = torch.zeros_like(rewards)
			advantages = torch.zeros_like(rewards)
			gae = 0.0
			next_value = 0.0

			for t in reversed(range(len(rewards))):
				delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
				gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
				advantages[t] = gae
				next_value = values[t]

		returns = advantages + values

		# Normalize advantages
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		dataset_size = len(obs)
		indices = np.arange(dataset_size)

		# PPO update
		for _ in tqdm(range(self.ppo_epochs), desc="PPO Epoch", leave=False):
			np.random.shuffle(indices)

			for start in tqdm(range(0, dataset_size, self.batch_size), desc="Batches", leave=False):
				end = start + self.batch_size
				mb_idx = indices[start:end]

				# Convert numpy indices to torch long tensor for indexing
				mb_idx_torch = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

				# Slice minibatches
				mb_obs = obs[mb_idx_torch]
				mb_actions = actions[mb_idx_torch]
				mb_old_logprobs = old_logprobs[mb_idx_torch]
				mb_advantages = advantages[mb_idx_torch]

				# Forward pass
				mu = self.policy(mb_obs)
				dist = torch.distributions.Normal(mu, 0.1)
				new_logprobs = dist.log_prob(mb_actions).sum(-1)
				entropy = dist.entropy().sum(-1)

				# Ratio
				ratio = (new_logprobs - mb_old_logprobs).exp()

				# Surrogate Loss
				surr1 = ratio * mb_advantages
				surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages

				policy_loss = -torch.min(surr1, surr2).mean()
				entropy_loss = -self.entropy_coef * entropy.mean()

				loss = policy_loss + entropy_loss

				# Update
				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
				self.optimizer.step()

		# Update old policy
		self.policy_old.load_state_dict(self.policy.state_dict())
		self.policy_old.eval()

	# ------------------------------------------------------------------ #
	# 6. Reset at start of new episode
	# ------------------------------------------------------------------ #
	def reset_episode(self):
		self.prev_board = None
		self.buffer_obs.clear()
		self.buffer_actions.clear()
		self.buffer_logprobs.clear()
		self.buffer_rewards.clear()
		self.buffer_dones.clear()
		self.prev_obs = None