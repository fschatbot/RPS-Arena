from typing import Union
import random
from datetime import datetime
from math import pi, cos, sin
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import numpy as np

class GameUniverse:
	def __init__(self, strategies):
		self.strategies = strategies
		self.radius = 10
		self.width = 800
		self.height = 800

		self.config = {
			'amount': 300,
			'clustering': -1,
			'dpi': 200,
			'stochasticity': 0.01, # Amount of randomness in movement
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
			surroundings = self.objects[np.arange(self.config['amount']) != i]
			acc_x, acc_y = self.strategies[int(self.objects[i, 4])](self.objects[i], surroundings)
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
		positions = self.objects[:, :2]
		types = self.objects[:, 4].astype(int)
		diff = positions[:, None, :] - positions[None, :, :]
		dist_sq = np.sum(diff**2, axis=-1)
		distances = np.sqrt(dist_sq)

		mask = np.triu(distances <= self.radius, k=1)  # only upper triangle
		i_idx, j_idx = np.where(mask)
		for i, j in zip(i_idx, j_idx):
			type_a = types[i]
			type_b = types[j]
			if type_a == type_b: continue # Nothing will happen
			win_chance = 0.5
			if (type_a + 1) % 3 == type_b: win_chance -= self.config['win_edge']
			else: win_chance += self.config['win_edge']
			if random.random() < win_chance:
				# A wins
				types[j] = type_a
			else:
				# B wins
				types[i] = type_b
		self.objects[:, 4] = types
	
		self.steps += 1

	def is_victory(self) -> bool:
		# Can be sped up here by interloop termination
		unique_types = np.unique(self.objects[:, 4])
		return len(unique_types) == 1

	def draw(self, ax):
		ax.clear()
		# Scatter plot each object based on its type
		for obj_type in range(3):
			xs = self.objects[self.objects[:, 4] == obj_type, 0]
			ys = self.objects[self.objects[:, 4] == obj_type, 1]
			ax.scatter(xs, ys, c=['red', 'blue', 'green'][obj_type], label=['rock', 'paper', 'scissors'][obj_type], s=self.radius)
		# legend on top right
		ax.legend(loc='upper right')
		# ax.axis('off')
		ax.set_xlim(0, self.width)
		ax.set_ylim(0, self.height)

def chasing(current:np.ndarray, surroundings:np.ndarray):
	prey = surroundings[(surroundings[:, 4] == (current[4] + 1) % 3)] # Only look for preys
	vec = prey[:, :2] - current[:2] # Vectors to all prey
	mag = np.linalg.norm(vec, axis=1) # Distances to all objects
	best = np.argmin(mag) if len(mag) > 0 else None
	return vec[best] / mag[best] if best is not None else (0, 0) # Normalize to unit vector


def weighted_chasing(current, surroundings):
	weight_map = {current[4]: -1, (current[4] + 1) % 3: -2, (current[4] + 2) % 3: 2}
	diff = surroundings[:, :2] - current[:2]
	distances = np.linalg.norm(diff, axis=1)**2 + 1e-9
	diff /= distances[:, np.newaxis]
	# Apply weights
	weights = np.array([weight_map[int(obj[4])] for obj in surroundings])
	diff *= weights[:, np.newaxis]
	move = np.sum(diff, axis=0)
	mag = np.linalg.norm(move)
	return (move / mag) if mag > 0 else (0, 0)


universe = GameUniverse([weighted_chasing, weighted_chasing, weighted_chasing])
universe.populate()
progress = tqdm(desc="Simulating", unit="step")
while not universe.is_victory():
	universe.step()
	progress.update(1)
progress.close()
universe.populate()

fig, ax = plt.subplots(figsize=(universe.width / universe.config['dpi'], universe.height / universe.config['dpi']), dpi=universe.config['dpi'])
padding = 0.01
plt.subplots_adjust(left=padding, right=1-padding, top=1-padding, bottom=padding)
writer = FFMpegWriter(fps=24, bitrate=4000)
output_name = f"exports/simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

with writer.saving(fig, output_name, dpi=universe.config['dpi']):
	progress = tqdm(desc="Simulating", unit="step")
	while not universe.is_victory():
		universe.step()
		universe.draw(ax)
		progress.update(1)
		writer.grab_frame()
	progress.close()

plt.close(fig)
print(f"Exported: {output_name}")