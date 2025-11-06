from typing import Union
import random
from datetime import datetime
from math import pi, cos, sin
from matplotlib import pyplot as plt

class GameUniverse:
	def __init__(self, strategies):
		self.objects = [] # {'position': (x, y), 'velocity': (vx, vy), 'type': type} where type is 0: rock, 1: paper, 2: scissors
		self.strategies = strategies
		self.radius = 10
		self.width = 800
		self.height = 800

		self.config = {
			'amount': 300,
			'clustering': 40,
			'dpi': 200,
			'stochasticity': 0.01, # Amount of randomness in movement
			'win_edge': 0.5, # 0.5 + win_edge equals the chance for predator to win
			'edge_behavior': 'bounce', # 'wrap' or 'bounce'
			'max_speed': 5,
		}

		self.steps = 0

		# Sanity checks
		assert len(self.strategies) == 3, "There must be exactly three strategies"

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

			self.objects.append({
				'position': (min(max(x, self.radius), self.width - self.radius), min(max(y, self.radius), self.height - self.radius)),
				'velocity': (0, 0),
				'type': i % 3,
			})

	def step(self):
		assert len(self.objects) == self.config['amount'], "Object count mismatch"
		# Simulatenously calculate the decision of all objects based on their strategies
		updates = []
		for obj in self.objects:
			surroundings = [other for other in self.objects if other != obj]
			acc_x, acc_y = self.strategies[obj['type']](obj, surroundings)
			acc_x += (2 * random.random() - 1) * self.config['stochasticity']
			acc_y += (2 * random.random() - 1) * self.config['stochasticity']
			mag = (acc_x ** 2 + acc_y ** 2) ** 0.5
			if mag > 1:
				acc_x /= mag
				acc_y /= mag
			updates.append((acc_x, acc_y))
		
		# Apply the updates
		for (new_ax, new_ay), obj in zip(updates, self.objects):
			new_vx, new_vy = (obj['velocity'][0] + new_ax, obj['velocity'][1] + new_ay)
			# Limit speed
			speed = (new_vx ** 2 + new_vy ** 2) ** 0.5
			if speed > self.config['max_speed']:
				new_vx = (new_vx / speed) * self.config['max_speed']
				new_vy = (new_vy / speed) * self.config['max_speed']
			
			# Update position
			new_x, new_y = (obj['position'][0] + new_vx, obj['position'][1] + new_vy)

			# Handle edge behavior
			if self.config['edge_behavior'] == 'wrap':
				new_x = new_x % self.width
				new_y = new_y % self.height
			elif self.config['edge_behavior'] == 'bounce':
				if new_x < self.radius or new_x > self.width - self.radius:
					new_vx = -new_vx
					new_x = min(max(new_x, self.radius + 1e-6), self.width - self.radius - 1e-6)
				if new_y < self.radius or new_y > self.height - self.radius:
					new_vy = -new_vy
					new_y = min(max(new_y, self.radius + 1e-6), self.height - self.radius - 1e-6)
			
			obj['position'] = (new_x, new_y)
			obj['velocity'] = (new_vx, new_vy)
		
		# Handle interactions
		for i in range(len(self.objects)):
			for j in range(i + 1, len(self.objects)):
				obj_a = self.objects[i]
				obj_b = self.objects[j]
				if obj_a['type'] == obj_b['type']: continue # Nothing will happen
				dx = obj_a['position'][0] - obj_b['position'][0]
				dy = obj_a['position'][1] - obj_b['position'][1]
				distance = (dx ** 2 + dy ** 2) ** 0.5
				if distance > self.radius: continue

				win_chance = 0.5
				if (obj_a['type'] + 1) % 3 == obj_b['type']: win_chance -= self.config['win_edge']
				else: win_chance += self.config['win_edge']

				if random.random() < win_chance:
					# A wins
					obj_b['type'] = obj_a['type']
				else:
					# B wins
					obj_a['type'] = obj_b['type']
		
		self.steps += 1

	def draw(self):
		# Draw all of the objects as scatter plots and export them into the exports folder
		plt.figure(figsize=(self.width / self.config['dpi'], self.height / self.config['dpi']), dpi=self.config['dpi'])
		# Scatter plot each object based on its type
		for obj_type in range(3):
			xs = [obj['position'][0] for obj in self.objects if obj['type'] == obj_type]
			ys = [obj['position'][1] for obj in self.objects if obj['type'] == obj_type]
			plt.scatter(xs, ys, c=['red', 'blue', 'green'][obj_type], label=['rock', 'paper', 'scissors'][obj_type], s=self.radius)
		# legend on top right
		plt.legend(loc='upper right')
		plt.axis('off')
		plt.xlim(0, self.width)
		plt.ylim(0, self.height)
		padding = 0.01
		plt.subplots_adjust(left=padding, right=1-padding, top=1-padding, bottom=padding)
		plt.savefig(f"exports/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.steps}.png")
		plt.close()

def chasing(current, surroundings):
	# Implement chasing behavior
	best_target = None
	best_distance = float('inf')

	for obj in surroundings:
		if (obj['type'] + 1) % 3 != current['type']:  continue  # Only chase the type that current can defeat
		dx = obj['position'][0] - current['position'][0]
		dy = obj['position'][1] - current['position'][1]
		distance = (dx ** 2 + dy ** 2) ** 0.5
		if distance < best_distance:
			best_distance = distance
			best_target = (dx, dy)
	
	return best_target if best_target is not None else (0, 0) # Normalize to unit vector


def weighted_chasing(current, surroundings):
	# Implement weighted chasing behavior
	# +2 weight for prey, -1 weight for same type, -2 weight for predator
	move_x, move_y = 0, 0
	weight_map = {current['type']: 0, (current['type'] + 1) % 3: -2, (current['type'] + 2) % 3: 2}
	for obj in surroundings:
		dx = obj['position'][0] - current['position'][0]
		dy = obj['position'][1] - current['position'][1]
		distance = (dx ** 2 + dy ** 2) ** 0.5 + 1e-9  # Avoid division by zero
		weight = weight_map.get(obj['type'], 0) / distance
		move_x += weight * (dx / distance)
		move_y += weight * (dy / distance)

	mag = (move_x ** 2 + move_y ** 2) ** 0.5
	return (move_x / mag, move_y / mag) if mag > 0 else (0, 0)


uni = GameUniverse([chasing, weighted_chasing, weighted_chasing])
uni.populate()
uni.draw()
for _ in range(1_000):
	uni.step()
	uni.draw()