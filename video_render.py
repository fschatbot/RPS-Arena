from engine import GameUniverse
from strategies import njit_weighted_chasing, chasing, RLStrategy
import ffmpeg
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch.cuda
import os

os.makedirs("exports", exist_ok=True)
MAX_STEPS = 1500

# Create RLStrategy for type 0 (example). Place RLStrategy instance in the strategies list
rl_for_type0 = RLStrategy(controlled_type=0, use_pretrained=True)
rl_for_type0.import_model("pretrained_models/checkpoint_game_180.pth")

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
termination_reason = None

while not universe.is_victory():
	universe.step()
	progress.update(1)
	frame = universe.draw()
	process.stdin.write(np.asarray(frame).tobytes())
	
	if universe.steps >= MAX_STEPS:
		termination_reason = "Max steps reached"
		break
else:
	termination_reason = "Victory"

progress.close()
process.stdin.close()
process.wait()

universe.render_history()

print(f"Exported: {output_name}")
print(f"Termination conditions: {termination_reason}")
print(f"Final counts: Rock: {universe.history[-1][0]}, Paper: {universe.history[-1][1]}, Scissor: {universe.history[-1][2]}")