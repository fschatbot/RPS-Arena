# Technical Design Document

The project simulates a dynamic Rock–Paper–Scissors (RPS) environment where autonomous agents (entities) interact under configurable rules. The system follows an **Entity–Environment** architecture. The goal of the project is to study emergent behaviors, strategies, and dynamics in a competitive multi-agent setting.

# Module Plan

## Game Universe (class)

Contains code for rendering the world, importing configurations, simiulating the world, and exporting analytics and video.

Functions: `step()`, `populate()`, `draw()`, `render_history()`, `is_victory()`
Properties: `strategies`, `config`, `history`, `objects`, `step`

# Strategy Function

This function at each step and for each entity provided the current entity being processed and the board state. It must return a 2D vector representing the acceleration direction.

# Strategy Class

This class is to be used when information regarding the game needs to be stored over time. The class's `compute_actions(...)` returns the acceleration vector for all of its entities. The `compute_and_store_rewards(...)` function is called at the end of each step to assign rewards based on the state change. Finally, the `train_on_episode()` function is called at the end of each game to allow for any training or evolution of the strategy.

Functions: `compute_actions()`, `compute_and_store_rewards()`, `train_on_episode()`, `import_model()`, `export_model()`, `get_episode_reward()`
Properties: `controlled_type`, etc...

## Entity (Class)

Contains code for managing motion updates, collision resolution, and abstract `brain()` method for decision-making for each entity.

The brain function is to output a vector which would be normalized as unit vector before being added to the velocity.

The `brain()` function is calculated simultaneously for all entities at each time step before any position updates ensuring fair decision-making.

Function/Properties: `update_position()`, `brain()`, `change_type()`, `position`, `velocity`, `type`, `id`

## Implemented Strategies

1. **Chaser Strategy**: move toward nearest winnable opponent.
2. **Distance-Weighted Strategy**: move in the weighted direction $a = \sum_{i=1}^n \frac{x_i - x}{(\|x_i - x\| + \epsilon)^2}$ emphasizing nearby opponents more.
3. **Learning Strategy**: use a lightweight Reinforcement Learning to adapt movement based on rewards (wins/survivals).
4. **Evolutionary Strategy**: maintain a population of strategies, evolving them based on performance over episodes.

# **Technology Stack**

## **Languages and Tools**

- **Python 3.14+**
- **Markdown** (for documentation)
- **Command-line interface** for simulation runs

## **Libraries**

- **Matplotlib** – analytics and visualization
- **PyTorch** – Reinforcement learning models
- **Pillow** & **FFmpeg** – Image saving and manipulation
- **NumPy**, **SciPy**, **Numba** – numerical computations and optimizations

# **Updated Repository**

https://github.com/fschatbot/RPS-Arena

# How to run

1. Clone the repository.
2. Install the required libraries using pip:
   ```
   pip install -r requirements.txt
   ```
3. Run the simulation using the command line:
   ```
   python genetic_train.py # For evolutionary training
   OR
   python train.py # For RL training
   ```
4. Evaluate strategies using:
   ```
   python evaluate.py
   ```
5. Render videos using:
   ```
   python video_render.py
   ```
