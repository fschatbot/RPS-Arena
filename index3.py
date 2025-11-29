import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from math import pi, sqrt, cos, sin
from tqdm.auto import tqdm

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
WIDTH = 800
HEIGHT = 800
N_AGENTS = 300  # Total agents (will be split evenly 100/100/100)
MATCHES = 500
MAX_STEPS_PER_MATCH = 700  # Cap to prevent infinite loops
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor for RL

# Visibility Radius: Area = 10% of total map
# pi * r^2 = 0.1 * (W * H)  =>  r = sqrt(0.1 * W * H / pi)
VISIBILITY_RADIUS = sqrt((WIDTH * HEIGHT * 0.1) / pi)

# Create directory for analytics
os.makedirs("analytics_exports", exist_ok=True)

# ==========================================
# NEURAL NETWORK (THE BRAIN)
# ==========================================
class PolicyNetwork(nn.Module):
	def __init__(self, obs_dim=9, hidden_dim=64):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 2)  # Output: Velocity X, Velocity Y
		)

	def forward(self, x):
		# Normalize output to ensure movement direction is unit length (or zero)
		return F.normalize(self.net(x), dim=-1)

# ==========================================
# REWARD LOGIC
# ==========================================
def calculate_rewards(model_type, same_count, pred_count, prey_count):
	"""
	Computes reward based on the user-specified models.
	"""
	rewards = np.zeros_like(same_count, dtype=np.float32)

	if model_type == 1:
		# Model 1: +1 Same, -1 Pred, -1 Prey
		rewards += (same_count * 1.0)
		rewards += (pred_count * -1.0)
		rewards += (prey_count * -1.0)

	elif model_type == 2:
		# Model 2: +1 Same, -1 Pred.
		# IF Pred > 0: +1 Prey
		# IF Pred == 0: -1 Prey

		rewards += (same_count * 1.0)
		rewards += (pred_count * -1.0)

		# Mask where predators exist
		has_predators = (pred_count > 0)

		# Add prey reward where predators exist
		rewards[has_predators] += (prey_count[has_predators] * 1.0)

		# Subtract prey reward where NO predators exist
		rewards[~has_predators] += (prey_count[~has_predators] * -1.0)

	return rewards

# ==========================================
# SIMULATION ENGINE (THE UNIVERSE)
# ==========================================
class Universe:
	def __init__(self, width, height, n_agents):
		self.width = width
		self.height = height
		self.n_agents = n_agents
		self.radius = 10  # Physical size of agent

		# State: [x, y, vx, vy, type_id]
		# Type ID: 0=Rock, 1=Paper, 2=Scissor
		self.state = np.zeros((n_agents, 5), dtype=np.float32)

		# Initialize History for Analytics
		self.history = []

	def reset(self):
		self.history = []
		# Random positions
		self.state[:, 0] = np.random.uniform(0, self.width, self.n_agents)
		self.state[:, 1] = np.random.uniform(0, self.height, self.n_agents)
		self.state[:, 2:4] = 0  # Zero initial velocity

		# Assign Types evenly
		for i in range(3):
			self.state[i::3, 4] = i

	def get_population_counts(self):
		types = self.state[:, 4].astype(int)
		counts = [np.sum(types == 0), np.sum(types == 1), np.sum(types == 2)]
		return counts

	def step(self, actions):
		# Apply actions (accelerations/velocity changes)
		# actions shape: (N, 2)

		# Update Velocity
		self.state[:, 2:4] += actions * 0.5  # Acceleration factor

		# Cap Speed
		max_speed = 5.0
		speed = np.linalg.norm(self.state[:, 2:4], axis=1)
		mask = speed > max_speed
		self.state[mask, 2:4] = (self.state[mask, 2:4] / speed[mask][:, None]) * max_speed

		# Update Position
		self.state[:, 0:2] += self.state[:, 2:4]

		# Bounce off walls
		# X boundaries
		left_mask = self.state[:, 0] < 0
		right_mask = self.state[:, 0] > self.width
		self.state[left_mask, 2] *= -1
		self.state[right_mask, 2] *= -1
		self.state[:, 0] = np.clip(self.state[:, 0], 0, self.width)

		# Y boundaries
		top_mask = self.state[:, 1] < 0
		bot_mask = self.state[:, 1] > self.height
		self.state[top_mask, 3] *= -1
		self.state[bot_mask, 3] *= -1
		self.state[:, 1] = np.clip(self.state[:, 1], 0, self.height)

		# Handle Collisions & Conversions
		self._handle_interactions()

		# Record Stats
		self.history.append(self.get_population_counts())

	def _handle_interactions(self):
		# Use cKDTree for efficient collision detection
		positions = self.state[:, 0:2]
		tree = cKDTree(positions)

		# Query pairs within collision distance (2 * radius)
		pairs = tree.query_pairs(self.radius * 2)

		for i, j in pairs:
			type_i = int(self.state[i, 4])
			type_j = int(self.state[j, 4])

			# If different types, check RPS rules
			if type_i != type_j:
				# (0 vs 1) -> 1 wins (Paper covers Rock)
				# (1 vs 2) -> 2 wins (Scissor cuts Paper)
				# (2 vs 0) -> 0 wins (Rock smashes Scissor)

				# Check if i beats j
				if (type_i == 0 and type_j == 2) or \
				   (type_i == 1 and type_j == 0) or \
				   (type_i == 2 and type_j == 1):
					# i wins, j converts to i
					self.state[j, 4] = type_i
				else:
					# j wins, i converts to j
					self.state[i, 4] = type_j

	def get_observations_and_counts(self):
		"""
		Returns:
		1. Observation Tensor for RL (N, 9)
		2. Counts of (Same, Pred, Prey) for Reward Calculation
		"""
		positions = self.state[:, :2]
		types = self.state[:, 4].astype(int)
		velocities = self.state[:, 2:4]

		N = self.n_agents
		obs_list = []

		# KDTree for visibility queries
		tree = cKDTree(positions)

		# Arrays to store counts for reward calculation
		count_same = np.zeros(N)
		count_pred = np.zeros(N)
		count_prey = np.zeros(N)

		# Build observations per agent
		# Ideally this would be vectorized for speed, but loop is clearer for logic
		# Optimization: Query all neighbors within visibility radius
		indices_list = tree.query_ball_point(positions, VISIBILITY_RADIUS)

		# Pre-calculate type relations
		# prey_of[t] is the prey of t
		prey_map = {0: 2, 1: 0, 2: 1}
		pred_map = {0: 1, 1: 2, 2: 0}

		observations = np.zeros((N, 9), dtype=np.float32)

		for i, neighbors in enumerate(indices_list):
			my_type = types[i]
			my_pos = positions[i]

			# Neighbors includes self, remove self
			if i in neighbors:
				neighbors.remove(i)

			if not neighbors:
				# No one visible: [0,0,0 type_onehot, 0,0 closest_dx, 0,0 closest_dy]
				# Just set type onehot
				observations[i, my_type] = 1.0
				continue

			# Filter neighbors
			nb_indices = list(neighbors)
			nb_pos = positions[nb_indices]
			nb_types = types[nb_indices]

			# Vectors to neighbors
			diffs = nb_pos - my_pos
			dists = np.linalg.norm(diffs, axis=1)

			# Find closest enemy (Predator) and closest food (Prey)
			pred_type = pred_map[my_type]
			prey_type = prey_map[my_type]

			is_pred = (nb_types == pred_type)
			is_prey = (nb_types == prey_type)
			is_same = (nb_types == my_type)

			# Update counts for Reward System
			count_same[i] = np.sum(is_same)
			count_pred[i] = np.sum(is_pred)
			count_prey[i] = np.sum(is_prey)

			# Construct Observation State for RL
			# [Type(3), ClosestPrey(2), ClosestPred(2), CountPrey(1), CountPred(1)] (Normalized)

			# One-hot type
			observations[i, my_type] = 1.0

			# Closest Prey Vector
			if np.any(is_prey):
				closest_prey_idx = np.argmin(dists[is_prey])
				# We need the vector from the filtered list
				# Get the actual index in the 'diffs' array
				# is_prey is boolean mask on diffs
				prey_diffs = diffs[is_prey]
				v_prey = prey_diffs[closest_prey_idx]
				# Normalize
				v_prey_norm = v_prey / (np.linalg.norm(v_prey) + 1e-6)
				observations[i, 3:5] = v_prey_norm

			# Closest Pred Vector
			if np.any(is_pred):
				closest_pred_idx = np.argmin(dists[is_pred])
				pred_diffs = diffs[is_pred]
				v_pred = pred_diffs[closest_pred_idx]
				v_pred_norm = v_pred / (np.linalg.norm(v_pred) + 1e-6)
				observations[i, 5:7] = v_pred_norm

			# Counts (Normalized roughly)
			observations[i, 7] = count_prey[i] / 10.0
			observations[i, 8] = count_pred[i] / 10.0

		return observations, count_same, count_pred, count_prey

# ==========================================
# TRAINING MANAGER
# ==========================================
def train():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Running on device: {device}")
	print(f"Visibility Radius: {VISIBILITY_RADIUS:.2f}")
	
	# Initialize Universe
	universe = Universe(WIDTH, HEIGHT, N_AGENTS)
	
	# Initialize Policy
	policy = PolicyNetwork().to(device)
	optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
	
	# Select Reward Model (Change this to 1 or 2 as needed)
	CURRENT_MODEL = 2
	print(f"Using Reward Model: {CURRENT_MODEL}")
	
	# Analytics storage
	match_durations = []
	
	for match_idx in range(1, MATCHES + 1):
		universe.reset()
	
		# Buffers for Trajectory (Monte Carlo / REINFORCE style)
		# We collect log_probs and rewards for the whole match
		log_probs_batch = []
		rewards_batch = []
	
		step_count = 0
		no_contact_steps = 0  # Initialize stagnation counter
		pbar = tqdm(total=MAX_STEPS_PER_MATCH, desc=f"Match {match_idx}/{MATCHES}", leave=False)
	
		while step_count < MAX_STEPS_PER_MATCH:
			# 1. Observe
			obs_np, c_same, c_pred, c_prey = universe.get_observations_and_counts()
	
	
			if np.sum(c_pred) + np.sum(c_prey) == 0: no_contact_steps += 1
			else: no_contact_steps = 0
	
			if no_contact_steps > 100: break
	
			obs_t = torch.FloatTensor(obs_np).to(device)
	
			# 2. Act
			# Policy outputs direction (vx, vy)
			# We treat this as a deterministic vector for movement,
			# but for training we need a distribution?
			# Simplified: Use the output vector + noise for exploration, assume Gaussian.
			# Or standard Policy Gradient: Output mean, sample action.
	
			# Let's treat the network output as the mean of a Gaussian
			action_means = policy(obs_t)
			cov_mat = torch.eye(2).to(device) * 0.1 # Fixed variance
	
			# Sample actions
			dist = torch.distributions.MultivariateNormal(action_means, cov_mat)
			actions_sample = dist.sample()
			log_prob = dist.log_prob(actions_sample)
	
			actions_np = actions_sample.cpu().detach().numpy()
	
			# 3. Step Universe
			universe.step(actions_np)
	
			# 4. Calculate Step Reward
			# Note: We calculate reward based on the state *before* the move (observation)
			# or *after*? Usually after. Let's re-observe counts after step.
			_, next_c_same, next_c_pred, next_c_prey = universe.get_observations_and_counts()
	
			step_rewards = calculate_rewards(CURRENT_MODEL, next_c_same, next_c_pred, next_c_prey)
	
			# Store
			log_probs_batch.append(log_prob)
			rewards_batch.append(torch.FloatTensor(step_rewards).to(device))
	
			# Check Victory
			counts = universe.get_population_counts()
			if counts.count(0) >= 2: # Only one type left (or zero)
				break
			
			step_count += 1
			pbar.update(1)
	
		pbar.close()
		match_durations.append(step_count)
	
		# ==========================================
		# EXPORT ANALYTICS CHART
		# ==========================================
		history = np.array(universe.history)
		if len(history) > 0:
			plt.figure(figsize=(10, 6))
			plt.plot(history[:, 0], color='gray', label='Rock', linewidth=2)
			plt.plot(history[:, 1], color='#f0f0f0', label='Paper', linestyle='--', linewidth=2) # Paper is white-ish
			plt.plot(history[:, 1], color='black', alpha=0.3, linewidth=1) # Make Paper visible on white bg
			plt.plot(history[:, 2], color='red', label='Scissor', linewidth=2)
	
			plt.title(f"Match {match_idx} Population Over Time (Steps: {step_count})")
			plt.xlabel("Time Steps")
			plt.ylabel("Population")
			plt.legend()
			plt.grid(True, alpha=0.3)
			plt.savefig(f"analytics_exports/match_{match_idx}_population.png")
			plt.close()
	
		# ==========================================
		# TRAIN (POLICY UPDATE)
		# ==========================================
		# We have a list of (N_Agents) log_probs and (N_Agents) rewards per step.
		# Structure: Steps -> Agents
	
		# Flatten: We treat every agent's step as an independent sample for the batch
		# shape: (Steps, Agents)
	
		if len(log_probs_batch) > 0:
			batch_log_probs = torch.stack(log_probs_batch) # (Steps, Agents)
			batch_rewards = torch.stack(rewards_batch)     # (Steps, Agents)
	
			# Normalize Rewards (Stability)
			# Calculate returns (simple sum for this match, or discounted)
			# Since matches are short and chaotic, we can just use immediate rewards or short horizon.
			# Let's use Discounted Returns
			R = 0
			returns = []
			for r in reversed(batch_rewards):
				R = r + GAMMA * R
				returns.insert(0, R)
			returns = torch.stack(returns)
	
			# Normalize
			returns = (returns - returns.mean()) / (returns.std() + 1e-9)
	
			# Policy Gradient Loss: -mean(log_prob * return)
			loss = -(batch_log_probs * returns).mean()
	
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	
			print(f"Match {match_idx} Complete. Duration: {step_count}. Loss: {loss.item():.4f}")
		else:
			print(f"Match {match_idx} Complete (Instant End). No training.")
	
		if match_idx % 10 == 0:
			model_save_path = os.path.join("analytics_exports", f"policy_model_{match_idx}.pth")
			torch.save(policy.state_dict(), model_save_path)
			print(f"Model saved to {model_save_path}")