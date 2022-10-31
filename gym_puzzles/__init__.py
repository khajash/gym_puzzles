from gym.envs.registration import register

register(
    id='MultiRobotPuzzle-v0',
    entry_point='gym_puzzles.envs:MultiRobotPuzzle',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzleHeavy-v0',
    entry_point='gym_puzzles.envs:MultiRobotPuzzleHeavy',
    max_episode_steps=3000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzle-v2',
    entry_point='gym_puzzles.envs:MultiRobotPuzzle2',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzleHeavy-v2',
    entry_point='gym_puzzles.envs:MultiRobotPuzzleHeavy2',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzle-v3',
    entry_point='gym_puzzles.envs:RobotPuzzleBase',
    max_episode_steps=1500,
    reward_threshold=100,
)