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

# ------------------------------------------------------------------
# Heuristic Strategies (Opponents)
# ------------------------------------------------------------------

@njit
def njit_weighted_chasing(current: np.ndarray, surroundings: np.ndarray, i: int):
    cur_type = int(current[4])
    N = surroundings.shape[0]
    weights = np.empty(N, dtype=np.float64)
    for j in range(N):
        t = int(surroundings[j, 4])
        d = (t - cur_type) % 3
        if d == 0:
            weights[j] = -0.5 # Slight repulsion from allies
        elif d == 1:
            weights[j] = -2.0 # Strong repulsion from predators
        else:
            weights[j] = 2.0  # Strong attraction to prey
            
    diff = surroundings[:, :2] - current[:2]
    
    # Simple inverse distance weighting
    for j in range(N):
        dx = diff[j, 0]; dy = diff[j, 1]
        dist2 = dx * dx + dy * dy + 1e-9
        # Normalize direction and weight by distance (closer = stronger)
        factor = weights[j] / dist2
        diff[j, 0] = dx * factor
        diff[j, 1] = dy * factor

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

# ------------------------------------------------------------------
# Neural Network Modules
# ------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(128, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim) - 0.5) # Initialize with small variance

    def forward(self, obs):
        h = self.net(obs)
        mu = torch.tanh(self.mu_head(h)) # Action range [-1, 1]
        logstd = self.logstd.expand_as(mu)
        return mu, logstd

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs)

# ------------------------------------------------------------------
# PPO Strategy
# ------------------------------------------------------------------

class RLStrategy:
    """
    A Reinforcement Learning agent implementing the PPO (Proximal Policy Optimization) algorithm.
    Includes fixes for:
    1. GPS Bug (Translation Invariance)
    2. Reward Scaling & Credit Assignment
    3. Proper Batching & Normalization
    """

    def __init__(self, controlled_type: int = 0, use_pretrained: bool = False):
        self.controlled_type = controlled_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dimensions
        self.obs_dim = 86 # 10 neighbors * 8 features + 2 self velocity + 4 wall feats
        self.act_dim = 2

        # Networks
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)
        
        # Hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip_ratio = 0.2
        self.ppo_epochs = 10     # More epochs for sample efficiency
        self.batch_size = 2048   # Large batch size for stability
        self.entropy_coef = 0.01 # Encourage exploration
        self.max_grad_norm = 0.5
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Buffers
        self.current_active = {} # Tracks trajectories of currently living agents
        self.finished_buffer = [] # Stores completed trajectories waiting for training
        self.games_counter = 0    # Tracks how many games have passed
        self.prev_state = None    # For calculating shaping rewards (deltas)

        if use_pretrained:
            self.import_model("pretrained_models/BEST_MODEL_ROCK.pth")

    def _make_full_board_obs(self, board: np.ndarray) -> torch.Tensor:
        """
        Generates observations with correct translation invariance.
        Each agent sees the world relative to itself.
        """
        my_indices = np.where(board[:, 4] == self.controlled_type)[0]
        if len(my_indices) == 0:
            return torch.empty((0, self.obs_dim), device=self.device)
        
        # Normalize global inputs
        all_pos = board[:, :2] / np.array([BOARD_WIDTH, BOARD_HEIGHT])
        all_vel = board[:, 2:4] / MAX_SPEED
        all_types = board[:, 4].astype(int)
        
        my_type = self.controlled_type
        prey_type = (my_type + 2) % 3
        pred_type = (my_type + 1) % 3
        
        observations = []

        for idx in my_indices:
            my_p = all_pos[idx]
            
            # --- FIX: Translation Invariance ---
            # Calculate relative position of everyone else to me
            rel_pos = all_pos - my_p 
            
            # CRITICAL FIX: Ensure 'self' is strictly 0,0. 
            # (Mathematically it is, but we ensure no floating point garbage)
            rel_pos[idx] = 0.0 
            
            # Calculate distances
            dists = np.linalg.norm(rel_pos, axis=1)
            
            # Identify types
            is_ally = (all_types == my_type).astype(np.float32)
            is_prey = (all_types == prey_type).astype(np.float32)
            is_pred = (all_types == pred_type).astype(np.float32)
            
            # Construct feature matrix for all agents
            # Shape: (N_AGENTS, 8) -> [rx, ry, dist, vx, vy, ally, prey, pred]
            feats = np.column_stack([
                rel_pos,                # 2
                dists,                  # 1
                all_vel,                # 2
                is_ally,                # 1
                is_prey,                # 1
                is_pred                 # 1
            ])

            # --- K-Nearest Neighbors ---
            # We only look at the 10 closest agents (excluding self usually, but here we sort)
            # We sort by distance. Index 0 is self (dist 0).
            sorted_indices = np.argsort(dists)
            
            # Take closest 11 (Self + 10 neighbors)
            # We remove self (index 0) from the neighbors list to give 10 external neighbors
            closest_indices = sorted_indices[1:11] 
            
            feats_near = feats[closest_indices]
            
            # Padding if agents died and < 10 exist
            num_near = feats_near.shape[0]
            if num_near < 10:
                pad = np.zeros((10 - num_near, 8))
                feats_near = np.vstack([feats_near, pad])
                
            # --- Wall Features ---
            # Distance to 4 walls: Left, Right, Top, Bottom
            # my_p is [x, y] normalized 0-1
            wall_feats = np.array([
                my_p[0], 1.0 - my_p[0], # Dist to Left, Right
                my_p[1], 1.0 - my_p[1]  # Dist to Top, Bottom
            ])

            # Flatten and concat
            # 10 neighbors * 8 feats = 80
            # + 2 self velocity = 82
            # + 4 wall feats = 86
            obs_vec = np.concatenate([
                feats_near.flatten(), 
                all_vel[idx], 
                wall_feats
            ])
            
            observations.append(obs_vec)
            
        return torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)

    def compute_actions(self, board_state: np.ndarray) -> np.ndarray:
        my_indices = np.where(board_state[:, 4] == self.controlled_type)[0]
        if len(my_indices) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # 1. Generate Observations
        obs = self._make_full_board_obs(board_state)
        self.prev_state = board_state.copy() # Store for reward calc
        
        # 2. Query Network
        with torch.no_grad():
            mu, logstd = self.actor(obs)
            std = torch.exp(logstd)
            dist = Normal(mu, std)
            acts = dist.sample()
            logps = dist.log_prob(acts).sum(dim=-1)
            vals = self.critic(obs).squeeze(-1)
            
        actions = acts.cpu().numpy()
        
        # 3. Store Experience (Trajectory Initialization)
        for j, agent_id in enumerate(my_indices):
            if agent_id not in self.current_active:
                self.current_active[agent_id] = {
                    'obs': [], 'acts': [], 'logps': [], 'vals': [], 'rews': [], 'dones': []
                }
            
            traj = self.current_active[agent_id]
            traj['obs'].append(obs[j].clone())
            traj['acts'].append(acts[j].clone())
            traj['logps'].append(logps[j].clone())
            traj['vals'].append(vals[j].clone())
            
        return actions

    def compute_and_store_rewards(self, after_state: np.ndarray):
        if self.prev_state is None:
            return
            
        before = self.prev_state
        after_types = after_state[:, 4].astype(int)
        
        # Pre-compute positions for shaping
        my_type = self.controlled_type
        prey_type = (my_type + 2) % 3
        pred_type = (my_type + 1) % 3
        
        # Get positions of interest
        prey_pos = after_state[after_state[:, 4] == prey_type][:, :2]
        pred_pos = after_state[after_state[:, 4] == pred_type][:, :2]
        
        # Iterate over agents that WERE active
        # We need to use list(keys) because we might delete from dict
        for agent_id in list(self.current_active.keys()):
            traj = self.current_active[agent_id]
            
            # --- 1. Check Survival (Terminal State) ---
            is_dead = False
            # If agent_id is no longer my type, it died (or became something else)
            if after_types[agent_id] != my_type:
                is_dead = True
                
            # --- 2. Calculate Individual Reward ---
            step_reward = -0.01 # Small time penalty to encourage action
            
            if is_dead:
                step_reward = -1.0 # High penalty for death
            else:
                # --- 3. Shaping Reward (Distance) ---
                # Only if alive
                my_p = after_state[agent_id, :2]
                prev_p = before[agent_id, :2]
                
                # A. Reward for approaching Prey
                if len(prey_pos) > 0:
                    # Dist to nearest prey NOW
                    dists_new = np.linalg.norm(prey_pos - my_p, axis=1)
                    min_dist_new = np.min(dists_new)
                    
                    # Dist to nearest prey BEFORE
                    # Note: We use PREVIOUS prey positions for strict correctness, 
                    # but using current prey positions for both is an acceptable approximation 
                    # for "am I closer to them now than I was a second ago"
                    dists_old = np.linalg.norm(prey_pos - prev_p, axis=1)
                    min_dist_old = np.min(dists_old)
                    
                    # Positive if got closer (dist decreased)
                    step_reward += (min_dist_old - min_dist_new) * 0.5

                # B. Reward for avoiding Predator
                if len(pred_pos) > 0:
                    dists_new = np.linalg.norm(pred_pos - my_p, axis=1)
                    min_dist_new = np.min(dists_new)
                    
                    dists_old = np.linalg.norm(pred_pos - prev_p, axis=1)
                    min_dist_old = np.min(dists_old)
                    
                    # If I am very close to predator, punish hard
                    if min_dist_new < 60.0: # 3x radius
                         step_reward -= 0.05
                    
                    # Reward for moving away (dist increased)
                    step_reward += (min_dist_new - min_dist_old) * 0.5
            
            # Store reward and done
            traj['rews'].append(step_reward)
            traj['dones'].append(is_dead)
            
            # --- 4. Handle Termination ---
            if is_dead:
                # Calculate Advantage for this trajectory immediately
                self._finish_trajectory(traj, last_val=0.0) # Value of death is 0
                self.finished_buffer.append(traj)
                del self.current_active[agent_id]
                
        # (Newborn agents are handled automatically in compute_actions via "if not in current_active")

    def _finish_trajectory(self, traj, last_val=0.0):
        """
        Computes GAE and Returns for a finished trajectory.
        """
        rews = torch.tensor(traj['rews'], dtype=torch.float32, device=self.device)
        vals = torch.stack(traj['vals'])
        
        T = len(rews)
        # if len(vals) == T - 1:
        #     vals = torch.cat([vals, torch.tensor([last_val], device=self.device)])
        
        adv = torch.zeros(T, device=self.device)
        ret = torch.zeros(T, device=self.device)
        gae = 0.0
        
        # GAE Loop (Backwards)
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = last_val
            else:
                next_val = vals[t+1].item()
                
            delta = rews[t] + self.gamma * next_val - vals[t].item()
            gae = delta + self.gamma * self.lmbda * gae
            adv[t] = gae
            ret[t] = adv[t] + vals[t].item()
            
        traj['advantages'] = adv
        traj['returns'] = ret
        
        # Clean up lists to save memory (we only need tensors now)
        del traj['rews']
        del traj['vals']
        del traj['dones']

    def train_on_episode(self):
        """
        Called at the end of a game.
        1. Finishes all active agents (truncation).
        2. Checks if we have enough games (5) to update.
        3. If yes, runs PPO update.
        """
        
        # 1. Truncate active trajectories (Game Over)
        # Value of end state is estimated by Critic
        for agent_id, traj in self.current_active.items():
            if len(traj['obs']) == 0: continue
            
            last_obs = traj['obs'][-1]
            with torch.no_grad():
                last_val = self.critic(last_obs.unsqueeze(0)).item()
                
            self._finish_trajectory(traj, last_val=last_val)
            self.finished_buffer.append(traj)
            
        self.current_active = {} # Clear active dict
        self.games_counter += 1
        
        # 2. Check update condition (Every 5 games)
        if self.games_counter < 5:
            return 
            
        # 3. PPO Update
        # print(f"Training on {len(self.finished_buffer)} trajectories...")
        
        # Consolidate all data into one big batch
        all_obs = torch.cat([torch.stack(t['obs']) for t in self.finished_buffer])
        all_acts = torch.cat([torch.stack(t['acts']) for t in self.finished_buffer])
        all_logps = torch.cat([torch.stack(t['logps']) for t in self.finished_buffer])
        all_advs = torch.cat([t['advantages'] for t in self.finished_buffer])
        all_rets = torch.cat([t['returns'] for t in self.finished_buffer])
        
        # Clear buffer
        self.finished_buffer = []
        self.games_counter = 0
        
        # Normalize Advantages (CRITICAL for stability)
        all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-8)
        
        # PPO Epochs
        dataset_size = all_obs.size(0)
        indices = np.arange(dataset_size)
        
        for _ in tqdm(range(self.ppo_epochs), desc="PPO Epoch", unit="epoch", leave=False):
            np.random.shuffle(indices)
            
            for start in tqdm(range(0, dataset_size, self.batch_size), desc="Batches", unit="batch", leave=False):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                b_obs = all_obs[batch_idx]
                b_acts = all_acts[batch_idx]
                b_logps_old = all_logps[batch_idx]
                b_advs = all_advs[batch_idx]
                b_rets = all_rets[batch_idx]
                
                # --- Actor Loss ---
                mu, logstd = self.actor(b_obs)
                std = torch.exp(logstd)
                dist = Normal(mu, std)
                
                new_logps = dist.log_prob(b_acts).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(new_logps - b_logps_old)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advs
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # --- Critic Loss ---
                vals = self.critic(b_obs).squeeze(-1)
                critic_loss = ((vals - b_rets) ** 2).mean()
                
                # Update
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

    def get_episode_reward(self):
        # Return dummy or aggregated metric (not used for logic, just display)
        return np.array([0.0])

    def import_model(self, path: str):
        if not os.path.exists(path):
            return
        save_dict = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(save_dict['actor'])
        self.critic.load_state_dict(save_dict['critic'])
        self.actor_optim.load_state_dict(save_dict['actor_optim'])
        self.critic_optim.load_state_dict(save_dict['critic_optim'])
        print(f"Loaded {path}")

    def export_model(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, path)