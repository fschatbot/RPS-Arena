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
        self.logstd = nn.Parameter(torch.zeros(act_dim) - 0.5) 

    def forward(self, obs):
        h = self.net(obs)
        mu = torch.tanh(self.mu_head(h)) 
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

@njit
def movement(observation: np.ndarray, weights: np.ndarray) -> tuple:
    weight = weights[:, observation[:, 2].astype(np.int32)]
    dx = np.sum(observation[:, 0] * weight)
    dy = np.sum(observation[:, 1] * weight)
    mag = (dx * dx + dy * dy) ** 0.5
    
    return (dx / mag, dy / mag) if mag > 0.0 else (0.0, 0.0)

def runner(weights: np.ndarray) -> callable:
    def strategy(current: np.ndarray, surroundings: np.ndarray, i: int):
        # Convert to observation format
        my_p = current[:2]
        rel_pos = surroundings[:, :2] - my_p 
        dists = (rel_pos[:, 0] * rel_pos[:, 0]) + (rel_pos[:, 1] * rel_pos[:, 1])
        rel_pos /= np.sqrt(dists)[:, np.newaxis] + 1e-9
        sorted_indices = np.argsort(dists)
        types = surroundings[:, 4].astype(int)
        types = (types - current[4]) % 3

        # Context-sensitive strategy
        if not np.any(types == 1): # No predator detected
            # Run to nearest prey
            prey_indices = np.where(types[sorted_indices] == 2)[0]
            prey_pos = rel_pos[sorted_indices[prey_indices[0]]]
            return prey_pos[0], prey_pos[1]


        feats = np.empty((surroundings.shape[0], 3), dtype=np.float32)
        feats[:, 0] = rel_pos[:, 0]
        feats[:, 1] = rel_pos[:, 1]
        feats[:, 2] = types
        sorted_feats = feats[sorted_indices]
        # Return the final movement
        return movement(sorted_feats, weights)
    
    return strategy


class GeneticStrategy:
    def __init__(self, controlled_type: int = 0, population_size: int = 50):
        self.controlled_type = controlled_type
        self.population_size = population_size
        self.population = []
        self.score = np.zeros(population_size, dtype=np.float32)
        # Each population member is a 3x300 numpy array of weights
        # Each column corresponds to an object type (0: rock, 1: paper, 2: scissor)
        # Each row corresponds to the weight for that type at that rank
        for _ in range(population_size):
            weights = np.random.randn(3, 300).astype(np.float32)
            self.population.append(weights)
        
        # Set the first individual to a known good strategy
        self.population[0][0] = -0.5
        self.population[0][1] = 2.0
        self.population[0][2] = -2.0
        
        self.gameid = 0
        self.meCount = []
    
    def observation(self, board: np.ndarray) -> np.ndarray:
        # Make a self-centered observation
        my_indices = np.where(board[:, 4] == self.controlled_type)[0]
        observations = []

        for idx in my_indices:
            my_p = board[idx, :2]
            rel_pos = board[:, :2] - my_p 
            dists = np.linalg.norm(rel_pos, axis=1)
            # Sort rel_pos by distance
            sorted_indices = np.argsort(dists)
            # Feature List (rel_pos_x, rel_pos_y, type) sorted by distance
            feats = np.column_stack([
                rel_pos,
                board[:, 4].astype(int)
            ])
            sorted_feats = feats[sorted_indices]
            observations.append(sorted_feats)
        
        return np.array(observations)
    
    def compute_actions(self, board_state):
        my_indices = np.where(board_state[:, 4] == self.controlled_type)[0]
        if len(my_indices) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        obs = self.observation(board_state)
        actions = []
        strategy = self.population[self.gameid]

        for i in range(len(my_indices)):
            act = movement(obs[i], strategy)
            actions.append(act)
        
        return np.array(actions, dtype=np.float32)
    
    def compute_and_store_rewards(self, after_state: np.ndarray):
        self.meCount.append(np.sum(after_state[:, 4] == self.controlled_type))
    
    def train_on_episode(self):
        # self.score[self.gameid] = self.meCount[-1]
        # self.score[self.gameid] = np.sum(self.meCount)
        self.score[self.gameid] = np.mean(self.meCount)
        self.meCount = []

        self.gameid += 1
        if self.gameid >= self.population_size:
            # Evolve population
            sorted_indices = np.argsort(self.score)[::-1]
            self.export_model(f"pretrained_models/score_{self.score[sorted_indices[0]]:,.2f}_gen.pth")
            tqdm.write(f"Best score in generation: {self.score[sorted_indices[0]]}")
            top_half = [self.population[i] for i in sorted_indices[:self.population_size // 2]]
            new_population = top_half.copy()
            while len(new_population) < self.population_size:
                parent = top_half[np.random.randint(len(top_half))]
                child = parent.copy()
                mutation = np.random.randn(*child.shape).astype(np.float32) * 0.01
                child += mutation
                new_population.append(child)
            self.population = new_population
            self.score = np.zeros(self.population_size, dtype=np.float32)
            self.gameid = 0
    
    def get_episode_reward(self):
        return np.array([0.0])

    def export_model(self, path: str):
        # Export the model of the best individual
        best_idx = np.argmax(self.score)
        best_weights = self.population[best_idx]
        np.save(path, best_weights)
    
    def import_model(self, path: str):
        if not os.path.exists(path):
            return
        best_weights = np.load(path)
        self.population[0] = best_weights
        print(f"Loaded {path}")



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
        
        
        self.obs_dim = (50*8 + 9) 
        self.act_dim = 2

        
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)
        
        
        self.lr = 1e-4
        self.gamma = 0.99
        self.lmbda = 0.95
        self.clip_ratio = 0.2
        self.ppo_epochs = 4     
        self.batch_size = 2048   
        self.entropy_coef = 0.1 
        self.max_grad_norm = 0.5
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        
        self.current_active = {} 
        self.finished_buffer = [] 
        self.games_counter = 0    
        self.prev_state = None    

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
        
        
        all_pos = board[:, :2] / np.array([BOARD_WIDTH, BOARD_HEIGHT])
        all_vel = board[:, 2:4] / MAX_SPEED
        all_types = board[:, 4].astype(int)
        
        my_type = self.controlled_type
        prey_type = (my_type + 2) % 3
        pred_type = (my_type + 1) % 3
        
        observations = []

        for idx in my_indices:
            my_p = all_pos[idx]
            rel_pos = all_pos - my_p 
            
            
            
            rel_pos[idx] = 0.0 
            
            
            dists = np.linalg.norm(rel_pos, axis=1)
            
            
            is_ally = (all_types == my_type).astype(np.float32)
            is_prey = (all_types == prey_type).astype(np.float32)
            is_pred = (all_types == pred_type).astype(np.float32)
            
            
            
            feats = np.column_stack([
                rel_pos,                
                dists,                  
                all_vel,                
                is_ally,                
                is_prey,                
                is_pred                 
            ])

            
            
            
            sorted_indices = np.argsort(dists)
            
            
            
            closest_indices = sorted_indices[1:51] 
            
            feats_near = feats[closest_indices]
            
            
            num_near = feats_near.shape[0]
            if num_near < 50:
                pad = np.zeros((50 - num_near, 8))
                feats_near = np.vstack([feats_near, pad])
                
            
            
            
            wall_feats = np.array([
                my_p[0], 1.0 - my_p[0], 
                my_p[1], 1.0 - my_p[1]  
            ])
            obs_vec = np.concatenate([
                feats_near.flatten(), 
                all_vel[idx], 
                wall_feats,
                [np.sum(is_ally), np.sum(is_prey), np.sum(is_pred)]
            ])
            
            observations.append(obs_vec)
            
        return torch.tensor(np.array(observations), dtype=torch.float32, device=self.device)

    def compute_actions(self, board_state: np.ndarray) -> np.ndarray:
        my_indices = np.where(board_state[:, 4] == self.controlled_type)[0]
        if len(my_indices) == 0:
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
        
        actions = np.clip(actions, -1.0, 1.0)
        return actions

    def compute_and_store_rewards(self, after_state: np.ndarray):
        if self.prev_state is None:
            return
            
        before = self.prev_state
        after_types = after_state[:, 4].astype(int)
        
        
        my_type = self.controlled_type
        prey_type = (my_type + 2) % 3
        pred_type = (my_type + 1) % 3
        
        
        prey_pos = after_state[after_state[:, 4] == prey_type][:, :2]
        pred_pos = after_state[after_state[:, 4] == pred_type][:, :2]
        
        
        
        for agent_id in list(self.current_active.keys()):
            traj = self.current_active[agent_id]
            
            
            is_dead = False
            
            if after_types[agent_id] != my_type:
                is_dead = True
                
            
            step_reward = -0.01 
            
            if is_dead:
                step_reward = -1.0 
            else:
                my_p = after_state[agent_id, :2]
                prev_p = before[agent_id, :2]
                
                if len(prey_pos) > 0:                    
                    dists_new = np.linalg.norm(prey_pos - my_p, axis=1)
                    min_dist_new = np.min(dists_new)
                    dists_old = np.linalg.norm(prey_pos - prev_p, axis=1)
                    min_dist_old = np.min(dists_old)
                    step_reward += (min_dist_old - min_dist_new) * 0.5
                if len(pred_pos) > 0:
                    dists_new = np.linalg.norm(pred_pos - my_p, axis=1)
                    min_dist_new = np.min(dists_new)
                    dists_old = np.linalg.norm(pred_pos - prev_p, axis=1)
                    min_dist_old = np.min(dists_old)
                    if min_dist_new < 60.0: 
                         step_reward -= 0.05
                    step_reward += (min_dist_new - min_dist_old) * 0.5
                
                # Give reward based on population advantage: (me - pred - prey) / total
                me_total = np.sum(after_types == my_type)
                pred_total = np.sum(after_types == pred_type)
                prey_total = np.sum(after_types == prey_type)
                step_reward += (me_total - pred_total - prey_total) / (me_total + pred_total + prey_total) * 0.1
                # Small award is pred_total < prey_total
                if pred_total < prey_total: step_reward += 0.05   
            
            traj['rews'].append(step_reward)
            traj['dones'].append(is_dead)
            
            
            if is_dead:
                
                self._finish_trajectory(traj, last_val=0.0) 
                self.finished_buffer.append(traj)
                del self.current_active[agent_id]
                

    def _finish_trajectory(self, traj, last_val=0.0):
        """
        Computes GAE and Returns for a finished trajectory.
        """
        rews = torch.tensor(traj['rews'], dtype=torch.float32, device=self.device)
        vals = torch.stack(traj['vals'])
        
        T = len(rews)
        
        
        
        adv = torch.zeros(T, device=self.device)
        ret = torch.zeros(T, device=self.device)
        gae = 0.0
        
        
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
        
        
        
        for agent_id, traj in self.current_active.items():
            if len(traj['obs']) == 0: continue
            
            last_obs = traj['obs'][-1]
            with torch.no_grad():
                last_val = self.critic(last_obs.unsqueeze(0)).item()
                
            self._finish_trajectory(traj, last_val=last_val)
            self.finished_buffer.append(traj)
            
        self.current_active = {} 
        self.games_counter += 1
        
        
        if self.games_counter < 10: return 
            
        
        
        
        
        all_obs = torch.cat([torch.stack(t['obs']) for t in self.finished_buffer])
        all_acts = torch.cat([torch.stack(t['acts']) for t in self.finished_buffer])
        all_logps = torch.cat([torch.stack(t['logps']) for t in self.finished_buffer])
        all_advs = torch.cat([t['advantages'] for t in self.finished_buffer])
        all_rets = torch.cat([t['returns'] for t in self.finished_buffer])
        
        
        self.finished_buffer = []
        self.games_counter = 0
        
        
        all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-8)
        
        
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
                
                
                mu, logstd = self.actor(b_obs)
                std = torch.exp(logstd)
                dist = Normal(mu, std)
                
                new_logps = dist.log_prob(b_acts).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                ratio = torch.exp(new_logps - b_logps_old)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_advs
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                
                vals = self.critic(b_obs).squeeze(-1)
                critic_loss = ((vals - b_rets) ** 2).mean()
                
                
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

    def get_episode_reward(self):
        
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