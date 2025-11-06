# Technical Design Document
 - outline your system architecture, main components, and data flow.

# Module Plan
 - break your project into logical modules or functions and describe their roles.
## Game Universe: (class)
This class contains the code for rendering the screen window, elements, and the graphs that would be required for the analysis. It is also the class that contains information regarding the universe like size of the box, wall interaction mechanics, entities present in the game, and the dimensions the game is gonna be played in.It is also the class that would house the visibility logic such as visibility radius and the amount of visible elements for each entity.

It also houses a 
## Entity: (class)
This class contains the logic for each indivdiual element. It will house the simiulate function which would update the position, velocity, accerlation, collision logic, and much more. It also contains an abstract function called brain which outputs an accerlation vector (normalized using $\frac{a}{||a||}$).

## Stratergy: (class) [inheriting Entity]
This class defines the brain function with 3 primary strategies (as of writing):

1. move towards the nearest entity against whom the current entity wins

2. move in a weighted direction based on distance and position of other entities: $
 a = \sum_{i=1}^n \frac{x_i}{||x_i - x|| + \epsilon}
 $

3. Reinforcement learning based strategy using a simple neural network.

# Technology Stack - list key libraries, frameworks, or APIs you plan to use.

## Tech Stack:
- Python
- Command Prompt
- Jyupter
- Markdown

## Libraries:
- PyGame
- Matplotlib
- PyTorch
- Pillow
- Pickle

# Updated Repository
