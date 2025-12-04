import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import random
from math import pi, cos, sin


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
			'clustering': -1,
			'dpi': 200,
			'stochasticity': 0.00,
			'win_edge': 0.5,
			'edge_behavior': 'bounce',
			'max_speed': 5,
		}
		self.objects = np.zeros((self.config['amount'], 5), dtype=np.float32)  # [x,y,vx,vy,type]
		self.steps = 0
		self.history = [] # Saves the population history

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
		random.seed(42)
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
		
		self.steps = 0
		self.history = [(self.config['amount'] // 3, self.config['amount'] // 3, self.config['amount'] // 3)]

	def step(self):
		assert self.objects.shape == (self.config['amount'], 5)
		N = self.objects.shape[0]

		# Prepare a container for updates (accelerations)
		updates = np.zeros((N, 2), dtype=np.float32)

		# If any strategy slots are RLStrategy, compute actions in batch per RLStrategy instance
		for t in range(3):
			strat = self.strategies[t]
			if hasattr(strat, 'compute_actions'):
				updates[self.objects[:, 4] == t] = strat.compute_actions(self.objects)

		# Compute updates per-entity:
		for i in range(N):
			t = int(self.objects[i, 4])
			strat = self.strategies[t]
			if not hasattr(strat, 'compute_actions'):
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
			if hasattr(strat, 'compute_and_store_rewards'):
				strat.compute_and_store_rewards(self.objects)

		self.steps += 1
		self.history.append((np.sum(self.objects[:, 4] == 0),
							 np.sum(self.objects[:, 4] == 1),
							 np.sum(self.objects[:, 4] == 2)))

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
	
	def render_history(self, path: str = None, extra: np.array = None):
		history_array = np.array(self.history)
		steps = np.arange(history_array.shape[0])

		plt.figure(figsize=(10, 6))
		plt.plot(steps, history_array[:, 0], label='Rock', color='blue')
		plt.plot(steps, history_array[:, 1], label='Paper', color='orange')
		plt.plot(steps, history_array[:, 2], label='Scissor', color='green')

		# Draw the extra on a different axis if provided
		if extra is not None:
			ax1 = plt.gca()
			ax2 = ax1.twinx()
			ax2.plot(np.arange(len(extra)), np.cumsum(extra), label='RL Reward', color='red', linestyle='--')
			ax2.set_ylabel('RL Reward')
			ax2.legend(loc='upper left')

		plt.xlabel('Steps')
		plt.ylabel('Population Count')
		plt.title('Population History Over Time')
		plt.legend()
		plt.grid()
		if path: plt.savefig(path)
		else: plt.show()
		plt.close()