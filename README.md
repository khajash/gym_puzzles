# gym_puzzles

This repository contains custom environments following the OpenAI Gym structure. Each environment contains multiple agents (although they are all centrally controlled by the same algorithm with the same observation and reward), whose goal is to move the T-shaped block to the goal location, marked by a circle. 

By adding this repository to your OpenAI Gym folder, you can simply use `gym.make("MultiRobotPuzzle-v0")` to call the environment.

### multi_robot_puzzle_00.py
In this environment, the agents have holonomic control, meaning they can move freely in the x and y-dimensions and rotate. 

![Centralized MultiRobot Puzzle 00](https://github.com/khajash/gym_puzzles/blob/master/EnvImages/CentralizedMRP0-light.jpg)


![Centralized MultiRobot Puzzle 00 Heavy](https://github.com/khajash/gym_puzzles/blob/master/EnvImages/CentralizedMRP0-Heavy.jpg)

![Centralized MultiRobot Puzzle 02](https://github.com/khajash/gym_puzzles/blob/master/EnvImages/CentralizedMRP1-HumanVision.jpg)

![Centralized MultiRobot Puzzle 02-AgentVision](https://github.com/khajash/gym_puzzles/blob/master/EnvImages/CentralizedMRP1-AgentVision.jpg)
