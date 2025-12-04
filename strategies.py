import numpy as np
from numba import njit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
from tqdm import tqdm

BOARD_WIDTH = 800
BOARD_HEIGHT = 800
MAX_SPEED = 5.0
N_AGENTS = 300


def chasing(current: np.ndarray, surroundings: np.ndarray, i):
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
    if cur_type == 1: return 0.0, 0.0  # Paper does not move
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

def weighted_chasing(current: np.ndarray, surroundings: np.ndarray, i: int):
    return njit_weighted_chasing(current, surroundings, i)

class Actor(nn.Module):
    def __init__(self, obs_dim=80, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        h = self.net(obs)
        mu = torch.tanh(self.mu_head(h))
        logstd = self.logstd.expand_as(mu)
        return mu, logstd

class Critic(nn.Module):
    def __init__(self, obs_dim=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs):
        return self.net(obs)

class RLStrategy:
    def __init__(self, controlled_type: int = 0, use_pretrained: bool = False):
        self.controlled_type = controlled_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = 80
        self.act_dim = 2
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.clip = 0.2
        self.gamma = 0.99
        self.lmbda = 0.95
        self.ppo_epochs = 4
        self.max_grad_norm = 0.5
        self.current_active = {}
        self.step_rewards = []
        self.prev_state = None
        if use_pretrained:
            self.import_model("pretrained_models/BEST_MODEL_ROCK.pth")  # Assume path

    def _make_full_board_obs(self, board: np.ndarray) -> torch.Tensor:
        my_indices = np.where(board[:, 4] == self.controlled_type)[0]
        if len(my_indices) == 0:
            return torch.empty((0, self.obs_dim), device=self.device)
        
        all_pos = board[:, :2] / np.array([BOARD_WIDTH, BOARD_HEIGHT])
        all_vel = board[:, 2:4] / MAX_SPEED
        all_types = board[:, 4].astype(int)
        observations = []
        
        my_type = self.controlled_type
        prey_type = (my_type + 2) % 3  # Prey beats me? Wait, rock(0) prey=scissors(2)=(0+2)%3, pred=paper(1)=(0+1)%3
        pred_type = (my_type + 1) % 3

        for idx in my_indices:
            my_p = all_pos[idx]
            rel_pos = all_pos - my_p
            rel_pos[idx] = my_p  # Self at origin? But dist=0 anyway
            
            dists = np.linalg.norm(rel_pos, axis=1, keepdims=True)
            
            is_ally = (all_types == my_type).astype(np.float32)
            is_prey = (all_types == prey_type).astype(np.float32)
            is_pred = (all_types == pred_type).astype(np.float32)
            
            feats = np.column_stack([
                rel_pos,
                dists,
                all_vel,  # Absolute vel; could relativize: all_vel - my_vel[idx]
                is_ally,
                is_prey,
                is_pred
            ])  # (N, 2+1+2+1+1+1=8)

            # Sort by distance
            sorted_indices = np.argsort(dists[:, 0])
            feats_near = feats[sorted_indices[:10]]

            # Pad to 10 if needed
            num_near = feats_near.shape[0]
            if num_near < 10:
                pad = np.zeros((10 - num_near, 8))
                feats_near = np.vstack([feats_near, pad])

            observations.append(feats_near.flatten())  # (80,)
        return torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)

    def compute_actions(self, board_state: np.ndarray) -> np.ndarray:
        my_indices = np.where(board_state[:, 4] == self.controlled_type)[0]
        num_my = len(my_indices)
        if num_my == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        obs = self._make_full_board_obs(board_state)
        self.prev_state = board_state.copy()
        
        with torch.no_grad():
            mu, logstd = self.actor(obs)
            std = torch.exp(logstd)
            dist = Normal(mu, std)
            acts = dist.sample()
            logps = dist.log_prob(acts).sum(dim=-1)
            vals = self.critic(obs).squeeze(-1)
        
        actions = acts.cpu().numpy()
        
        # Store pre-step data
        for j, i in enumerate(my_indices):
            if i not in self.current_active:
                self.current_active[i] = {
                    'obs': [], 'acts': [], 'logps': [], 'vals': [], 'rews': [], 'dones': [], 'agent_id': i
                }
            traj = self.current_active[i]
            traj['obs'].append(obs[j].clone().detach())
            traj['acts'].append(acts[j].clone().detach())
            traj['logps'].append(logps[j].clone().detach())
            traj['vals'].append(vals[j].clone().detach())
        
        return actions

    def compute_and_store_rewards(self, after_state: np.ndarray):
        if self.prev_state is None:
            return
        before = self.prev_state
        global_r = self._compute_reward(before, after_state)
        self.step_rewards.append(global_r)
        
        after_types = after_state[:, 4].astype(int)
        finished = []
        for i in list(self.current_active.keys()):
            traj = self.current_active[i]
            traj['rews'].append(float(global_r))
            traj['dones'].append(False)
            if after_types[i] != self.controlled_type:
                traj['dones'][-1] = True
                finished.append(traj)
                del self.current_active[i]
        
        # Newborn Rocks start empty traj
        for i in range(after_state.shape[0]):
            if after_types[i] == self.controlled_type and i not in self.current_active:
                self.current_active[i] = {
                    'obs': [], 'acts': [], 'logps': [], 'vals': [], 'rews': [], 'dones': [], 'agent_id': i
                }

    def get_episode_reward(self) -> np.array:
        return np.array(self.step_rewards)

    def train_on_episode(self):
        finished = []
        # Finish remaining active trajs (episode end)
        for traj in list(self.current_active.values()):
            num_steps = len(traj['obs'])
            if len(traj['dones']) < num_steps:
                traj['dones'] = [False] * num_steps
            if num_steps > 0:  # Ensure at least one step
                traj['dones'][-1] = True
            finished.append(traj)
        self.current_active = {}
        
        if not finished:
            self.step_rewards = []
            return
        
        for traj in finished:
            T = len(traj['rews'])
            if T == 0: continue
            
            # Convert lists to tensors
            rews_t = torch.tensor(traj['rews'], dtype=torch.float32, device=self.device)
            vals_t = torch.tensor(traj['vals'], dtype=torch.float32, device=self.device)
            
            # Handle the bootstrap value
            # If the episode ended naturally (died/won), value is 0.
            # If it was truncated (max steps), value is Critic(last_state).
            last_obs = traj['obs'][-1]
            last_done = traj['dones'][-1] # This should be True only if actually died
            
            if last_done:
                last_val = 0.0
            else:
                with torch.no_grad():
                    # unsqueeze to make batch size 1
                    last_val = self.critic(last_obs.unsqueeze(0)).item()
            
            adv = torch.zeros(T, device=self.device)
            ret = torch.zeros(T, device=self.device)
            gae_val = 0.0
            
            # Iterate backwards
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = last_val
                else:
                    next_val = vals_t[t+1].item()
                    
                delta = rews_t[t] + self.gamma * next_val - vals_t[t]
                gae_val = delta + self.gamma * self.lmbda * gae_val
                adv[t] = gae_val
                ret[t] = adv[t] + vals_t[t]
            
            traj['advantages'] = adv
            traj['returns'] = ret
        
        # Concat all data
        obs_all = torch.cat([torch.stack(traj['obs']) for traj in finished if len(traj['obs']) > 0])
        acts_all = torch.cat([torch.stack(traj['acts']) for traj in finished if len(traj['acts']) > 0])
        logps_old_all = torch.cat([torch.stack(traj['logps']) for traj in finished if len(traj['logps']) > 0])
        adv_all = torch.cat([traj['advantages'] for traj in finished if 'advantages' in traj])
        ret_all = torch.cat([traj['returns'] for traj in finished if 'returns' in traj])
        
        # Normalize adv
        adv_mean = adv_all.mean()
        adv_std = adv_all.std() + 1e-8
        adv_all = (adv_all - adv_mean) / adv_std
        
        total_steps = len(obs_all)
        if total_steps == 0:
            self.step_rewards = []
            return
        
        batch_size = min(64, total_steps)
        indices = torch.randperm(total_steps, device=self.device)
        
        for _ in tqdm(range(self.ppo_epochs), desc="PPO Epoch", unit="epoch", leave=False):
            for start in tqdm(range(0, total_steps, batch_size), desc="Batches", unit="batch", leave=False):
                end = start + batch_size
                mb_indices = indices[start:end]
                obs_mb = obs_all[mb_indices]
                acts_mb = acts_all[mb_indices]
                logps_old_mb = logps_old_all[mb_indices]
                adv_mb = adv_all[mb_indices]
                ret_mb = ret_all[mb_indices]
                
                # Actor update
                mu, logstd = self.actor(obs_mb)
                std = torch.exp(logstd)
                dist = Normal(mu, std)
                logp_new = dist.log_prob(acts_mb).sum(dim=-1)
                ratio = torch.exp(logp_new - logps_old_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                
                # Critic update
                val_new = self.critic(obs_mb).squeeze()
                val_loss = ((val_new - ret_mb) ** 2).mean()
                self.critic_optim.zero_grad()
                val_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
        
        self.step_rewards = []

    def _compute_reward(self, before: np.ndarray, after: np.ndarray) -> float:
        # Existing implementation unchanged
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

        r = 5.0 * pred_killed + 1.5 * prey_killed - 8.0 * my_died

        if len(np.unique(afterT)) == 1:
            r += 20000.0 if afterT[0] == my else -15000.0
            
        r += 2 * (my_a - pred_a - prey_a) / (my_a + pred_a + prey_a + 1e-8)

        my_positions = after[afterT == my][:, :2]
        
        if len(my_positions) > 0:
            prey_positions = after[afterT == prey][:, :2]
            if len(prey_positions) > 0:
                diff = my_positions[:, None, :] - prey_positions[None, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                min_dists = np.min(dists, axis=1)
                r -= 0.01 * np.mean(min_dists)

            pred_positions = after[afterT == pred][:, :2]
            if len(pred_positions) > 0:
                diff = my_positions[:, None, :] - pred_positions[None, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                min_dists = np.min(dists, axis=1)
                r += 0.01 * np.mean(min_dists)

        return float(r)
    
    def import_model(self, path: str):
        if not os.path.exists(path):
            print(f"Warning: Model {path} not found; starting from scratch.")
            return
        save_dict = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(save_dict['actor'])
        self.critic.load_state_dict(save_dict['critic'])
        self.actor_optim.load_state_dict(save_dict['actor_optim'])
        self.critic_optim.load_state_dict(save_dict['critic_optim'])
        print(f"Loaded model from {path}")

    def export_model(self, path: str):
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }
        torch.save(save_dict, path)
        