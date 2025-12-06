# train.py
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
import os
from engine import GameUniverse
from strategies import njit_weighted_chasing, runner
from joblib import Parallel, delayed

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
EVOLUTIONS = 500
MAX_STEPS_PER_GAME = 500
CONTROLLED_TYPE = 0
VALIDATION_GAMES = 3
OUTPUT_DIR = "pretrained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("analytics_exports", exist_ok=True)

opponent_strategy = njit_weighted_chasing


def run_game(brain:np.ndarray) -> int:
    strategy = runner(brain)
    universe = GameUniverse([
        strategy,
        opponent_strategy,
        opponent_strategy
    ])
    universe.populate()
    
    while not universe.is_victory() and universe.steps < MAX_STEPS_PER_GAME and universe.history[-1][0] > 0:
        universe.step()
    
    return float(universe.history[-1][0]) - float(len(universe.history) / MAX_STEPS_PER_GAME)  # Return number of remaining agents of controlled type

progress = tqdm(range(EVOLUTIONS), desc="Evolutions", unit="evolution", position=0)

population_size = 24
population = [np.random.randn(3, 300).astype(np.float32) for _ in range(population_size)]

# Source - https://stackoverflow.com/a/61900501
# Posted by user394430
# Retrieved 2025-12-05, License - CC BY-SA 4.0

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total, leave=False, dynamic_ncols=True, position=1) as self._pbar:
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


for evolution_idx in progress:
    scores = np.zeros(population_size)
    for _ in range(VALIDATION_GAMES):
        scores += np.array(ProgressParallel(n_jobs=-1, total=population_size)(delayed(run_game)(brain) for brain in population))
    scores = scores / VALIDATION_GAMES
    
    sorted_indices = np.argsort(scores)[::-1]
    progress.set_postfix({'best': scores[sorted_indices[0]], 'mean': np.mean(scores)})
    
    # Export the best model
    best_brain = population[sorted_indices[0]]
    np.save(os.path.join(OUTPUT_DIR, f"evolution_{evolution_idx+1}_score_{scores[sorted_indices[0]]:.2f}.npy"), best_brain)
    
    top_half = [population[i] for i in sorted_indices[:population_size // 2]]
    new_population = top_half.copy()
    while len(new_population) < population_size:
        parent = top_half[np.random.randint(len(top_half))]
        child = parent.copy()
        mutation = np.random.randn(*child.shape).astype(np.float32) * 0.01
        mutation_mask = np.random.rand(*child.shape) < (len(new_population) / population_size)
        child += mutation * mutation_mask
        new_population.append(child)
    population = new_population

