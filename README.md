# OpenAI Gym Puzzles

This repository contains custom OpenAI Gym environments using PyBox2D. Each environment contains multiple robots which are centrally controlled by a single agent, whose goal is to move the T-shaped block to the goal location, marked by a circle. For true mutli-agent environments check out [multiagent-env](https://github.com/khajash/multiagent-env).

## Setup

```
git clone https://github.com/khajash/gym_puzzles.git
cd gym_puzzles
python -m venv .env
source .env/bin/activate
pip install -e .
```

## Usage

```
import gym_puzzles
env = gym.make('MultiRobotPuzzle-v0')
```

## Envs Available
There are four envs currently configured:
-  **Holonomic Control**
    - `MultiRobotPuzzle-v0`
    - `MultiRobotPuzzleHeavy-v0`
- **Non-Holonomic Control**
    - `MultiRobotPuzzle-v2`
    - `MultiRobotPuzzleHeavy-v2`

#### `MultiRobotPuzzle-v0` and `MultiRobotPuzzleHeavy-v0`

In both of these environments, the agents have holonomic control, meaning they can move freely in the x and y-dimensions and rotate. 
The base version `MultiRobotPuzzle-v0` contains 2 agents with a lighter and smaller block than the heavy version.

<!-- ![Centralized MultiRobot Puzzle 00](./imgs/CentralizedMRP0-light.jpg) -->

<img src="./imgs/CentralizedMRP0-light.jpg" width="300">

The heavy version `MultiRobotPuzzleHeavy-v0` contains 5 agents with a block that is 2x the size of the block in the normal environment and 2x heavier.
<img src="./imgs/CentralizedMRP0-Heavy.jpg" width="300">

<!-- ![Centralized MultiRobot Puzzle 00 Heavy](./imgs/CentralizedMRP0-Heavy.jpg) -->


### `MultiRobotPuzzle-v2` and `MultiRobotPuzzleHeavy-v2`

In both of these environments, the agents have non-holonomic control, where agents control their linear velocity and turning radius, similar to a car. 
Both the base and heavy versions have the same size block, but the heavy version has a density significantly higher making it very difficult to move alone. This image shows the standard human vision rendering style showing solid fills for objects, vertices, and centroids.

<!-- ![Centralized MultiRobot Puzzle 02](./imgs/CentralizedMRP1-HumanVision1.jpg) -->
<img src="./imgs/CentralizedMRP1-HumanVision1.jpg" width="300">

*Rendering Agent Vision*

This rendering style is meant to give us an idea of what the agent sees. It only shows centroids, vertices, and vectors. 

<img src="./imgs/CentralizedMRP1-AgentVision.jpg" width="300">
<!-- ![Centralized MultiRobot Puzzle 02-AgentVision](./imgs/CentralizedMRP1-AgentVision.jpg) -->

