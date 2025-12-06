# train.py
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import torch
from engine import GameUniverse
from strategies import njit_weighted_chasing, RLStrategy, GeneticStrategy

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
N_GAMES = 1500
MAX_STEPS_PER_GAME = 500
CONTROLLED_TYPE = 0
OUTPUT_DIR = "pretrained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("analytics_exports", exist_ok=True)	

# ------------------------------------------------------------------
# Initialize RL agent (training mode)
# ------------------------------------------------------------------
rl_strategy = RLStrategy(controlled_type=CONTROLLED_TYPE, use_pretrained=False)
rl_strategy = GeneticStrategy(controlled_type=CONTROLLED_TYPE, population_size=24)

# Opponents: both use the strong weighted chasing
opponent_strategy = njit_weighted_chasing

# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
wins = 0
best_win_streak = 0
current_streak = 0
best_model_path = None

# ------------------------------------------------------------------
# Training loop: 500 full games
# ------------------------------------------------------------------
print(f"Starting training: {N_GAMES} games against njit_weighted_chasing")
progress = tqdm(range(N_GAMES), desc="Games", unit="game", position=0)


for game_idx in progress:
	# Create fresh universe

	for _ in range(4):
		universe = GameUniverse([
			rl_strategy,
			opponent_strategy,
			opponent_strategy
		])
		universe.populate()
		
		last_pop = rl_strategy.meCount.copy()

		# Play one full game
		gProg = tqdm(total=MAX_STEPS_PER_GAME, desc=" Steps", unit="step", leave=False, position=1)
		while not universe.is_victory() and universe.steps < MAX_STEPS_PER_GAME and universe.history[-1][0] > 0:
			universe.step()
			gProg.update(1)
		gProg.close()

		if universe.history[-1][0] == 0 or universe.history[-1][1] == 300:
			rl_strategy.meCount.extend([universe.history[-1][0]] * (MAX_STEPS_PER_GAME - universe.steps))

		rl_strategy.meCount = last_pop + [rl_strategy.meCount[-1]]

		universe.render_history(path=f"analytics_exports/game_{game_idx+1}_{_}.png")
	
	# Training
	# if (game_idx) % 24 == 0:
	# 	rl_strategy.export_model(f"{OUTPUT_DIR}/checkpoint_game_{game_idx+1}.pth")
	# 	progress.write(f"Checkpoint saved at game {game_idx+1}")
	rl_strategy.train_on_episode()

# ------------------------------------------------------------------
# Final save: export the last policy + best one
# ------------------------------------------------------------------
final_path = f"{OUTPUT_DIR}/final_trained_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
rl_strategy.export_model(final_path)

# Also copy the best model with a clear name
if best_model_path:
	import shutil
	shutil.copy(best_model_path, f"{OUTPUT_DIR}/BEST_MODEL_ROCK.pth")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Total games played : {game_idx + 1}")
print(f"Final win rate     : {wins / (game_idx + 1):.1%} ({wins}/{game_idx + 1})")
print(f"Best win streak    : {best_win_streak}")
print(f"Best model saved to: {best_model_path or 'N/A'}")
print(f"Final model saved to: {final_path}")
if best_model_path:
	print(f"Recommended for video: {OUTPUT_DIR}/BEST_MODEL_ROCK.pth")
print("="*60)