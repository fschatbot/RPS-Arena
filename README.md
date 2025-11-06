# Technical Design Document

The project simulates a dynamic Rock–Paper–Scissors (RPS) environment where autonomous agents (entities) interact under configurable rules. The system follows an **Entity–Environment** architecture. The goal of the project is to study emergent behaviors, strategies, and dynamics in a competitive multi-agent setting.

# Module Plan

## Game Universe (class)

Contains code for rendering the world, importing configurations, simiulating the world, and exporting analytics and video.

Functions/Properties: `step()`, `step_till_victory()`, `render()`, `export_video()`, `export_analytics()`, `find_closest()`, `width`, `height`, `entities`, `entity_radius`, `history`, `config`

## Entity (Class)

Contains code for managing motion updates, collision resolution, and abstract `brain()` method for decision-making for each entity.

The brain function is to output a vector which would be normalized as unit vector before being added to the velocity.

The `brain()` function is calculated simultaneously for all entities at each time step before any position updates ensuring fair decision-making.

Function/Properties: `update_position()`, `brain()`, `change_type()`, `position`, `velocity`, `type`, `id`

## Strategy (Subclass of Entity)

Implements the `brain()` logic:

1. **Chaser Strategy**: move toward nearest winnable opponent.
2. **Distance-Weighted Strategy**: move in the weighted direction $a = \sum_{i=1}^n \frac{x_i - x}{\|x_i - x\| + \epsilon}$ emphasizing nearby opponents more.
3. **Learning Strategy**: use a lightweight Reinforcement Learning to adapt movement based on rewards (wins/survivals).

Function/Properties: `brain()`

# **Technology Stack**

## **Languages and Tools**

- **Python 3.14+**
- **Jupyter Notebook** (for experiments and analysis)
- **Markdown** (for documentation)
- **Command-line interface** for simulation runs

## **Libraries**

- **PyGame** – rendering and simulation timing
- **Matplotlib** – analytics and visualization
- **PyTorch** – reinforcement learning models
- **Pillow** – image saving and manipulation
- **Pickle** – serialization of runs and configurations

# **Updated Repository**

https://github.com/fschatbot/RPS-Arena
