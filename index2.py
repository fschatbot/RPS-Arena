# index.py
# ATTEMPTING HIGH PRIORITY MANUVER
import psutil, os
p = psutil.Process(os.getpid())
try:
	p.nice(psutil.REALTIME_PRIORITY_CLASS)
	print('Process priority upgraded to Realtime!!')
except Exception:
	# Non-windows or privilege error: ignore
	pass

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

# --- PyTorch RL dependencies ---
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# RL: Policy, Obs, Rewards
# -------------------------
class RLPolicyNet(nn.Module):
    def __init__(self, obs_dim=9, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x):
        out = self.net(x)
        return F.normalize(out, dim=-1)

class RLStrategy:
    """
    Inference-only RL strategy:
    - Loads pre-trained model from 'analytics_exports/policy_model_500.pth'
    - compute_actions(positions_np, types_np) -> (N,2)
    - Training methods are disabled (No-op).
    """

    def __init__(self, controlled_type: int, device: Union[str, torch.device] = "cpu"):
        self.controlled_type = int(controlled_type)
        self.device = torch.device(device)
        self.model = RLPolicyNet().to(self.device)
        
        # --- MODIFICATION: Load Pre-trained Weights ---
        model_path = "analytics_exports/policy_model_500.pth"
        try:
            # map_location ensures it loads on CPU even if trained on CUDA
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Set to evaluation mode (disables dropout, etc.)
            print(f"RLStrategy (Type {controlled_type}): Successfully loaded model from {model_path}")
        except FileNotFoundError:
            print(f"WARNING: Model file '{model_path}' not found. Agents will act randomly.")
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")

        # No memory or training buffers needed for inference

    # -------------------------------------------------------------------------
    # OBSERVATION BUILDER (N,9)
    # -------------------------------------------------------------------------
    def _build_observations(self, positions: np.ndarray, types: np.ndarray):
        """
        Observation = [
            onehot(3),
            dx_to_prey, dy_to_prey,
            dx_to_pred, dy_to_pred,
            dist_to_prey, dist_to_pred
        ]
        """
        pos = positions.astype(np.float32)
        types = types.astype(np.int64)
        N = pos.shape[0]

        # pairwise diffs (N,N,2)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1) + 1e-9
        np.fill_diagonal(dist, 1e9)

        # prey/pred types
        prey_of = (types + 1) % 3
        pred_of = (types + 2) % 3

        # vectorized masks
        prey_mask = (types[None, :] == prey_of[:, None])
        pred_mask = (types[None, :] == pred_of[:, None])

        # nearest prey & pred
        prey_dist = np.where(prey_mask, dist, 1e9).min(axis=1)
        pred_dist = np.where(pred_mask, dist, 1e9).min(axis=1)

        prey_idx = np.where(prey_mask, dist, 1e9).argmin(axis=1)
        pred_idx = np.where(pred_mask, dist, 1e9).argmin(axis=1)

        # direction vectors
        idxs = np.arange(N)

        prey_vec = diff[idxs, prey_idx]
        pred_vec = diff[idxs, pred_idx]

        # one-hot
        onehot = np.zeros((N, 3), dtype=np.float32)
        onehot[np.arange(N), types] = 1.0

        obs = np.concatenate([
            onehot,
            prey_vec.astype(np.float32),
            pred_vec.astype(np.float32),
            prey_dist.reshape(-1, 1).astype(np.float32),
            pred_dist.reshape(-1, 1).astype(np.float32)
        ], axis=1)

        return obs  # (N,9)

    # -------------------------------------------------------------------------
    # ACTION COMPUTATION
    # -------------------------------------------------------------------------
    def compute_actions(self, positions: np.ndarray, types: np.ndarray):
        obs = self._build_observations(positions, types)
        obs_t = torch.from_numpy(obs).to(self.device)

        with torch.no_grad():
            actions_t = self.model(obs_t)        # (N,2)

        actions = actions_t.cpu().numpy().astype(np.float32)

        # Note: We no longer need to store _last_obs or _last_actions 
        # because we aren't calculating rewards.

        return actions

    # -------------------------------------------------------------------------
    # REWARD COMPUTATION (DISABLED)
    # -------------------------------------------------------------------------
    def compute_and_store_rewards(self, positions_after, types_after):
        # No-op for inference
        pass
    
    def train_step(self, batch_size=4096, gamma=0.99, lr=1e-3):
        # No-op for inference
        return 0.0

# -------------------------
# main
# -------------------------
def main():
	MAX_STEPS = 500

	# Create RLStrategy for type 0 (example). Place RLStrategy instance in the strategies list
	device = "cuda" if torch.cuda.is_available() else "cpu"
	rl_for_type0 = RLStrategy(controlled_type=0, device=device)

	# Example: type 0 -> RL, type 1 -> njit_weighted_chasing, type 2 -> njit_weighted_chasing
	universe = GameUniverse([rl_for_type0, njit_weighted_chasing, njit_weighted_chasing])
	universe.populate()

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
