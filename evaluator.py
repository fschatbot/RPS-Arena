# train.py
import numpy as np
from tqdm.auto import tqdm
import os
from engine import GameUniverse
from strategies import njit_weighted_chasing, runner
from joblib import Parallel, delayed

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
GAMES = 1000

opponent_strategy = njit_weighted_chasing


def run_game(strategy) -> int:
    universe = GameUniverse([
        strategy,
        opponent_strategy,
        opponent_strategy
    ])
    universe.populate()
    
    while not universe.is_victory() and universe.steps < 1500 and universe.history[-1][0] > 0:
        universe.step()
    
    # If stone no win then return 0, if timeout then return 0.5, if win return 1
    if universe.history[-1][0] > 0:
        return 1
    elif universe.steps >= 1500:
        return 0.5
    else:
        return 0

brain = np.load("pretrained_models/evolution_117_score_252.33.npy")
brain = np.load("pretrained_models/evolution_2_score_14.00.npy")
brain = np.load("pretrained_models/evolution_95_score_299.12.npy")
brain = np.load("pretrained_models/evolution_500_score_299.27.npy")
strategy = runner(brain)

# Source - https://stackoverflow.com/a/61900501
# Posted by user394430
# Retrieved 2025-12-05, License - CC BY-SA 4.0

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, leave=False, dynamic_ncols=True) as self._pbar:
            try:
                return Parallel.__call__(self, *args, **kwargs)
            finally:
                if self._use_tqdm:
                    self._pbar.clear()
                    self._pbar.close()

    def print_progress(self):
        if not self._use_tqdm or not hasattr(self, '_pbar'):
            return
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

print(f"Amount of parallel games: {os.cpu_count()}")
# We don't care whether the output comes in order or not
game_output = ProgressParallel(n_jobs=-1, total=GAMES)(delayed(run_game)(strategy) for _ in range(GAMES))
wins = np.sum(np.array(game_output) == 1)
print(f"Winrate over {GAMES} games: {wins / GAMES:.2%} [{wins} wins]")
print(f"Losses: {np.sum(np.array(game_output) == 0)}, Timeouts: {np.sum(np.array(game_output) == 0.5)}")