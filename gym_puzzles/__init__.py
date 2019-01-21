from gym.envs.registration import register

register(
    id='MultiRobotPuzzle-v0',
    entry_point='gym.envs.my_puzzles:MultiRobotPuzzle',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzleHeavy-v0',
    entry_point='gym.envs.my_puzzles:MultiRobotPuzzleHeavy',
    max_episode_steps=3000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzle-v2',
    entry_point='gym.envs.my_puzzles:MultiRobotPuzzle2',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzleHeavy-v2',
    entry_point='gym.envs.my_puzzles:MultiRobotPuzzleHeavy2',
    max_episode_steps=2000,
    reward_threshold=500,
)
