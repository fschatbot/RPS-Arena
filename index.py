# ATTEMPTING HIGH PRIORITY MANUVER
import psutil, os

p = psutil.Process(os.getpid())
p.nice(psutil.REALTIME_PRIORITY_CLASS)
print('Process priority upgraded to Realtime!!')

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

class GameUniverse:
	def __init__(self, strategies):
		self.strategies = strategies
		self.radius = 10
		self.width = 800
		self.height = 800

		self.config = {
			'amount': 300,
			'clustering': 40,
			'dpi': 200,
			'stochasticity': 0.00, # Amount of randomness in movement
			'win_edge': 0.5, # 0.5 + win_edge equals the chance for predator to win
			'edge_behavior': 'bounce', # 'wrap' or 'bounce'
			'max_speed': 5,
		}
		self.objects = np.zeros((self.config['amount'], 5)) # [x, y, vx, vy, type] where type is 0: rock, 1: paper, 2: scissors

		self.steps = 0

		# Sanity checks
		assert len(self.strategies) == 3, "There must be exactly three strategies"
		assert self.config['edge_behavior'] in ['wrap', 'bounce'], "Edge behavior must be 'wrap' or 'bounce'"
		assert self.config['amount'] > 0, "Amount must be positive"
		assert self.config['max_speed'] > 0, "Max speed must be positive"
		assert self.config['stochasticity'] >= 0, "Stochasticity must be non-negative"
		assert 0 <= self.config['win_edge'] <= 0.5, "Win edge must be between 0 and 0.5"
		assert self.width * self.height >= self.config['amount'] * (self.radius ** 2) * pi, "Area too small for the number of objects"

	def populate(self):
		assert self.config['amount'] % 3 == 0, "Amount must be divisible by 3"
		for i in range(self.config['amount']):
			if self.config['clustering'] > 0:
				cx, cy = self.width // 2, self.height // 2
				theta = (i % 3) * pi * 2 / 3 + pi / 2
				t_theta = min(self.width / (2 * abs(cos(theta)) + 1e-9), self.height / (2 * abs(sin(theta)) + 1e-9)) / 2

				# Find the center by taking the cx, cy and border coordinates
				x = cx + t_theta * cos(theta) + random.gauss(0, self.config['clustering'])
				y = cy + t_theta * sin(theta) + random.gauss(0, self.config['clustering'])
			else:
				x = random.randint(0, self.width)
				y = random.randint(0, self.height)

			x = min(max(x, self.radius), self.width - self.radius)
			y = min(max(y, self.radius), self.height - self.radius)
			self.objects[i] = [x, y, 0, 0, i % 3]

	def step(self):
		assert self.objects.shape == (self.config['amount'], 5), "Object count mismatch"
		# Simulatenously calculate the decision of all objects based on their strategies
		updates = np.zeros((self.config['amount'], 2)) # Acceleration updates
		for i in range(self.config['amount']):
			# Gather entities that are not self
			# surroundings = self.objects[np.arange(self.config['amount']) != i]
			acc_x, acc_y = self.strategies[int(self.objects[i, 4])](self.objects[i], self.objects, i)
			updates[i] = (acc_x, acc_y)
		
		# Add stochasticity to acceleration
		updates += np.random.uniform(-self.config['stochasticity'], self.config['stochasticity'], updates.shape)
		mag = np.linalg.norm(updates, axis=1)
		updates[mag > 1] /= mag[mag > 1][:, np.newaxis]

		# Apply acceleration to velocity
		self.objects[:, 2:4] += updates
		mag = np.linalg.norm(self.objects[:, 2:4], axis=1)
		self.objects[mag > self.config['max_speed'], 2:4] /= mag[mag > self.config['max_speed']][:, np.newaxis]

		# Apply velocity to position
		self.objects[:, :2] += self.objects[:, 2:4]
		# Handle edge behavior
		if self.config['edge_behavior'] == 'wrap':
			self.objects[:, 0] %= self.width
			self.objects[:, 1] %= self.height
		elif self.config['edge_behavior'] == 'bounce':
			# Multiply velocity by -1 if hitting edge
			self.objects[(self.objects[:, 0] < self.radius) | (self.objects[:, 0] > self.width - self.radius), 2] *= -1
			self.objects[(self.objects[:, 1] < self.radius) | (self.objects[:, 1] > self.height - self.radius), 3] *= -1
			# Correct position if out of bounds
			self.objects[:, 0] = np.clip(self.objects[:, 0], self.radius + 1e-9, self.width - self.radius - 1e-9)
			self.objects[:, 1] = np.clip(self.objects[:, 1], self.radius + 1e-9, self.height - self.radius - 1e-9)
		
		# Handle interactions
		tree = cKDTree(self.objects[:, :2])
		# Collision threshold is 2 * radius (two balls touch)
		collision_dist = 2.0 * self.radius
		pairs_set = tree.query_pairs(collision_dist)

		if len(pairs_set) > 0:
			pairs = np.array(list(pairs_set), dtype=np.intp)  # shape (M,2)
			i_idx = pairs[:, 0]
			j_idx = pairs[:, 1]

			pos = self.objects[:, :2]
			vel = self.objects[:, 2:4]
			types = self.objects[:, 4].astype(int)

			# --- Elastic collision resolution (equal mass) ---
			# For each colliding pair, resolve velocities and correct overlap
			for k in range(len(i_idx)):
				i = int(i_idx[k])
				j = int(j_idx[k])

				# vector from i -> j
				dx = pos[j, 0] - pos[i, 0]
				dy = pos[j, 1] - pos[i, 1]
				dist2 = dx * dx + dy * dy
				if dist2 == 0:
					# perfect overlap: jitter slightly to avoid division by zero
					# move j slightly
					dx = (np.random.random() - 0.5) * 1e-6
					dy = (np.random.random() - 0.5) * 1e-6
					dist2 = dx * dx + dy * dy

				dist = np.sqrt(dist2)
				# normal unit vector n (i -> j)
				nx = dx / (dist + 1e-12)
				ny = dy / (dist + 1e-12)

				# relative velocity v_i - v_j
				vix = vel[i, 0]
				viy = vel[i, 1]
				vjx = vel[j, 0]
				vjy = vel[j, 1]
				rvx = vix - vjx
				rvy = viy - vjy

				# velocity along normal
				rel_vel_norm = rvx * nx + rvy * ny

				# if rel_vel_norm > 0 they are moving apart -> still might overlap, so handle position correction but skip velocity swap
				if rel_vel_norm < 0:
					# equal-mass elastic collision: swap normal components (tangential stays)
					# project velocities onto normal and tangential directions
					# v_n_i = (v_i · n) n, v_t_i = v_i - v_n_i
					vni = (vix * nx + viy * ny)
					vnj = (vjx * nx + vjy * ny)

					# tangential components (vector)
					vti_x = vix - vni * nx
					vti_y = viy - vni * ny
					vtj_x = vjx - vnj * nx
					vtj_y = vjy - vnj * ny

					# swap normal scalars
					vni, vnj = vnj, vni

					# reconstruct velocities
					vel[i, 0] = vti_x + vni * nx
					vel[i, 1] = vti_y + vni * ny
					vel[j, 0] = vtj_x + vnj * nx
					vel[j, 1] = vtj_y + vnj * ny

				# --- positional correction to eliminate overlap ---
				# penetration depth = (2*radius - dist), move each ball by half along normal
				penetration = 2.0 * self.radius - dist
				if penetration > 0:
					# push them apart equally
					correction = 0.5 * penetration + 1e-9
					pos[i, 0] -= nx * correction
					pos[i, 1] -= ny * correction
					pos[j, 0] += nx * correction
					pos[j, 1] += ny * correction

			# --- Type fight resolution (after collisions) ---
			# compute vectorized type outcomes for all pairs
			a = types[i_idx]
			b = types[j_idx]

			win_chance = np.where(
				(a + 2) % 3 == b,                # a has advantage over b
				0.5 + self.config['win_edge'],
				0.5 - self.config['win_edge']
			)

			rolls = np.random.random(len(i_idx))
			a_wins = rolls < win_chance

			# where a wins -> j becomes a
			types[j_idx[a_wins]] = a[a_wins]
			# where a loses -> i becomes b
			types[i_idx[~a_wins]] = b[~a_wins]

			# write types back into objects
			self.objects[:, 4] = types
	
		self.steps += 1

	def is_victory(self) -> bool:
		# Can be sped up here by interloop termination
		unique_types = np.unique(self.objects[:, 4])
		return len(unique_types) == 1

	def draw(self):
		# Create blank RGB image
		img = Image.new("RGB", (self.width, self.height), (255, 255, 255))
		draw = ImageDraw.Draw(img)

		colors = {
			0: (255, 0, 0),     # rock = red
			1: (0, 0, 255),     # paper = blue
			2: (0, 255, 0)      # scissors = green
		}

		# Draw each object as a circle
		for obj in self.objects:
			x, y, _, _, obj_type = obj
			r = self.radius
			color = colors[obj_type]

			# Pillow circles use bounding box
			draw.ellipse(
				(x - r, y - r, x + r, y + r),
				fill=color,
				outline=None
			)

		return img   # return Pillow image

def chasing(current:np.ndarray, surroundings:np.ndarray, i):
	prey = surroundings[(surroundings[:, 4] == (current[4] + 1) % 3)] # Only look for preys
	vec = prey[:, :2] - current[:2] # Vectors to all prey
	mag = np.linalg.norm(vec, axis=1) # Distances to all objects
	best = np.argmin(mag) if len(mag) > 0 else None
	return vec[best] / (mag[best] + 1e-9) if best is not None else (0, 0) # Normalize to unit vector

@njit
def njit_weighted_chasing(current: np.ndarray, surroundings: np.ndarray, i: int):
	cur_type = int(current[4])
	N = surroundings.shape[0]

	# TYPE WEIGHTS:
	# d = (t - cur_type) % 3
	# d = 0 -> 0
	# d = 1 -> -2
	# d = 2 -> 2

	# allocate weights
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

	# compute diffs
	diff = surroundings[:, :2] - current[:2]

	# squared distances + epsilon
	# reassign in place to avoid temporary arrays (Numba-friendly)
	for j in range(N):
		dx = diff[j, 0]
		dy = diff[j, 1]
		dist2 = dx * dx + dy * dy + 1e-9
		diff[j, 0] = dx / dist2
		diff[j, 1] = dy / dist2

	# apply weights
	for j in range(N):
		w = weights[j]
		diff[j, 0] *= w
		diff[j, 1] *= w

	# sum
	move_x = 0.0
	move_y = 0.0
	for j in range(N):
		move_x += diff[j, 0]
		move_y += diff[j, 1]

	# normalize
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
	# Apply weights
	weights = np.array([weight_map[int(obj[4])] for obj in surroundings])
	diff *= weights[:, np.newaxis]
	move = np.sum(diff, axis=0)
	mag = np.linalg.norm(move)
	return (move / mag) if mag > 0 else (0, 0)

def main():
	MAX_STEPS = 5_000
	universe = GameUniverse([njit_weighted_chasing, njit_weighted_chasing, njit_weighted_chasing])
	universe.populate()
	# progress = tqdm(desc="Simulating", unit="step", total=MAX_STEPS)
	# while not universe.is_victory():
	# 	universe.step()
	# 	progress.update(1)
	# 	if universe.steps >= MAX_STEPS:
	# 		print("Reached maximum steps without victory. Exiting.")
	# 		break
	# progress.close()
	# universe.populate()

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
		if universe.steps >= MAX_STEPS:
			print("Reached maximum steps without victory. Exiting.")
			break
	
	progress.close()
	process.stdin.close()
	process.wait()

	print(f"Exported: {output_name}")


import cProfile
if __name__ == "__main__":
	cProfile.run('main()', 'profile_stats.prof')