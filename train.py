# train.py
import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import torch
from engine import GameUniverse
from strategies import njit_weighted_chasing, RLStrategy

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
N_GAMES = 500
MAX_STEPS_PER_GAME = 500
CONTROLLED_TYPE = 0
OUTPUT_DIR = "pretrained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Initialize RL agent (training mode)
# ------------------------------------------------------------------
rl_strategy = RLStrategy(controlled_type=CONTROLLED_TYPE, use_pretrained=False)

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
	universe = GameUniverse([
		rl_strategy,
		opponent_strategy,
		opponent_strategy
	])
	universe.populate()

	# Play one full game
	step = 0
	gProg = tqdm(total=MAX_STEPS_PER_GAME, desc=" Steps", unit="step", leave=False, position=1)
	while not universe.is_victory() and step < MAX_STEPS_PER_GAME and universe.history[-1][0] > 0:
		universe.step()
		step += 1
		gProg.update(1)
	gProg.close()
	rewards = rl_strategy.get_episode_reward()
	universe.render_history(path=f"analytics_exports/game_{game_idx+1}.png", extra=rewards)
	
	# Training
	rl_strategy.train_on_episode() # Train every 3 games
	if (game_idx + 1) % 10 == 0:
		rl_strategy.export_model(f"{OUTPUT_DIR}/checkpoint_game_{game_idx+1}.pth")
		progress.write(f"Checkpoint saved at game {game_idx+1}")

	# Determine winner
	final_types = universe.objects[:, 4].astype(int)
	unique_types = np.unique(final_types)
	winner = unique_types[0] if len(unique_types) == 1 else -1

	is_win = (winner == CONTROLLED_TYPE)

	if is_win:
		wins += 1
		current_streak += 1
		if current_streak > best_win_streak:
			best_win_streak = current_streak
			# Save best model so far
			path = f"{OUTPUT_DIR}/best_ever.pth"
			rl_strategy.export_model(path)
			best_model_path = path
	else:
		current_streak = 0

	# Update progress bar
	win_rate = wins / (game_idx + 1)
	progress.set_postfix({
		"WinRate": f"{win_rate:.3f}",
		"Streak": current_streak,
		"BestStreak": best_win_streak,
		"reward": f"{np.sum(rewards):.2f}"
	})

	# Optional: early stop if we're clearly dominating
	if win_rate >= 0.98 and game_idx >= 100:
		print(f"\nReached 98%+ win rate at game {game_idx+1}. Stopping early.")
		break

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