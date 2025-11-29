import numpy as np
from numba import njit

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

        self.net = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        ).to(self.device)

    def forward(self, x):
        # x: (B, 300, 5) → (B, 5, 300)
        x = x.permute(0, 2, 1)
        return self.net(x)  # (B, 2)


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
        self.episode_transitions: List[Transition] = []
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

    # ------------------------------------------------------------------ #
    # 2. Action selection
    # ------------------------------------------------------------------ #
    def compute_actions(self, board_state: np.ndarray) -> np.ndarray:
        obs = self._make_egocentric_obs(board_state)
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

        if self.prev_board is None:
            self.prev_board = after_state.copy()
            return

        reward = self._compute_reward(self.prev_board, after_state)
        done = len(np.unique(after_state[:, 4])) == 1

        # Get observations and actions from previous step
        prev_obs = self._make_egocentric_obs(self.prev_board)
        my_mask = (self.prev_board[:, 4] == self.controlled_type)

        if my_mask.any():
            my_prev_obs = prev_obs[my_mask]
            with torch.no_grad():
                mu = self.policy_old(my_prev_obs)
                dist = torch.distributions.Normal(mu, 0.1)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(-1)

            # Same action/logprob for all my agents
            for i, idx in enumerate(np.where(my_mask)[0]):
                self.episode_transitions.append(Transition(
                    obs=my_prev_obs[i:i+1],
                    action=action[i:i+1],
                    logprob=logprob[i:i+1],
                    reward=reward,
                    done=done
                ))

        self.prev_board = after_state.copy()

    # ------------------------------------------------------------------ #
    # 4. Asymmetric reward: predator death = jackpot
    # ------------------------------------------------------------------ #
    def _compute_reward(self, before: np.ndarray, after: np.ndarray) -> float:
        my = self.controlled_type
        pred = (my + 1) % 3
        prey = (my + 2) % 3

        my_b = np.sum(before[:, 4] == my)
        pred_b = np.sum(before[:, 4] == pred)
        prey_b = np.sum(before[:, 4] == prey)

        my_a = np.sum(after[:, 4] == my)
        pred_a = np.sum(after[:, 4] == pred)
        prey_a = np.sum(after[:, 4] == prey)

        pred_killed = pred_b - pred_a
        prey_killed = prey_b - prey_a
        my_died = my_b - my_a

        r = 5.0 * pred_killed + \
            1.5 * prey_killed - \
            8.0 * my_died

        # Terminal bonus
        if len(np.unique(after[:, 4])) == 1:
            r += 20000.0 if after[0, 4] == my else -15000.0

        # Dense shaping
        total = len(after)
        r += 25.0 * (my_a - my_b) / total
        r -= 40.0 * pred_a / total

        return float(r)

    # ------------------------------------------------------------------ #
    # 5. Full PPO update after each episode
    # ------------------------------------------------------------------ #
    def _train_on_episode(self):
        if len(self.episode_transitions) == 0:
            return

        batch = Transition(*zip(*self.episode_transitions))
        obs = torch.cat(batch.obs).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        old_logprobs = torch.cat(batch.logprob).to(self.device)
        rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        dones = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

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

        # PPO update
        for _ in range(self.ppo_epochs):
            mu = self.policy(obs)
            dist = torch.distributions.Normal(mu, 0.1)
            logprobs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)

            ratio = (logprobs - old_logprobs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.entropy_coef * entropy.mean()

            loss = policy_loss + entropy_loss

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
        self.episode_transitions.clear()