# gym_puzzles

This repository contains custom environments following the OpenAI Gym structure. Each environment contains multiple agents (although they are all centrally controlled by the same algorithm with the same observation and reward), whose goal is to move the T-shaped block to the goal location, marked by a circle. Check out [multiagent-env](https://github.com/khajash/multiagent-env) repo for true multiagent environment.

By adding this repository to your OpenAI Gym folder, you can simply use `gym.make("MultiRobotPuzzle-v0")` to call the environment. Environment names are:
- `MultiRobotPuzzle-v0`
- `MultiRobotPuzzleHeavy-v0`
- `MultiRobotPuzzle-v2`
- `MultiRobotPuzzleHeavy-v2`

### multi_robot_puzzle_00.py

In this environment, the agents have holonomic control, meaning they can move freely in the x and y-dimensions and rotate. 

*MultiRobotPuzzle-v0*

This env contains 2 agents with a lighter and smaller block than the heavy version.

![Centralized MultiRobot Puzzle 00](./EnvImages/CentralizedMRP0-light.jpg)

*MultiRobotPuzzleHeavy-v0*

This env contains 5 agents with a block that is 2x the size of the block in the normal environment and 2x heavier.

![Centralized MultiRobot Puzzle 00 Heavy](./EnvImages/CentralizedMRP0-Heavy.jpg)


### multi_robot_puzzle_02.py

In this environment, the agents have non-holonomic control, where agents control their linear velocity and turning radius, similar to a car. 

*MultiRobotPuzzle-v2 or MultiRobotPuzzleHeavy-v2*

Both the normal and heavy versions have the same size block, but heavy has a density significantly higher making it very difficult to move alone. This image shows the standard human vision rendering style showing solid fills for objects, vertices, and centroids.

![Centralized MultiRobot Puzzle 02](./EnvImages/CentralizedMRP1-HumanVision1.jpg)

*Rendering Agent Vision*

This rendering style is meant to give us an idea of what the agent sees. It only shows centroids, vertices, and vectors. 

![Centralized MultiRobot Puzzle 02-AgentVision](./EnvImages/CentralizedMRP1-AgentVision.jpg)

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