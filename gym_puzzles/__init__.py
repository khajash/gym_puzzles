from gym.envs.registration import register

register(
    id='RobotPuzzle-v0',
    entry_point='my_gym.envs:RobotPuzzle',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='RobotPuzzle-v1',
    entry_point='my_gym.envs:RobotPuzzle2',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='RobotPuzzle-v2',
    entry_point='my_gym.envs:RobotPuzzle3',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='RobotPuzzle-v3',
    entry_point='my_gym.envs:RobotPuzzle4',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='RobotPuzzle-v4',
    entry_point='my_gym.envs:RobotPuzzle5',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='RobotPuzzle-v5',
    entry_point='my_gym.envs:RobotPuzzle5Unitize',
    max_episode_steps=2000,
    reward_threshold=500,
)


register(
    id='MultiRobotPuzzle-v0',
    entry_point='my_gym.envs:MultiRobotPuzzle',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzleHeavy-v0',
    entry_point='my_gym.envs:MultiRobotPuzzleHeavy',
    max_episode_steps=3000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzle-v1',
    entry_point='my_gym.envs:MultiRobotPuzzle1',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzle-v2',
    entry_point='my_gym.envs:MultiRobotPuzzle2',
    max_episode_steps=2000,
    reward_threshold=500,
)

register(
    id='MultiRobotPuzzleHeavy-v2',
    entry_point='my_gym.envs:MultiRobotPuzzleHeavy2',
    max_episode_steps=2000,
    reward_threshold=500,
)



register(
    id='Puzzle-v0',
    entry_point='my_gym.envs:Puzzle',
    max_episode_steps=400,
    reward_threshold=400,
)


register(
    id='SimplePuzzle-v0',
    entry_point='my_gym.envs:SimplePuzzle',
    max_episode_steps=800,
    reward_threshold=500,
)

register(
    id='SimplePuzzlePixels-v0',
    entry_point='my_gym.envs:SimplePuzzlePixels',
    max_episode_steps=800,
    reward_threshold=500,
)

register(
    id='SimplePuzzleHighRes-v0',
    entry_point='my_gym.envs:SimplePuzzleHighRes',
    max_episode_steps=800,
    reward_threshold=500,
)

register(
    id='SimplePuzzleHighResPixels-v0',
    entry_point='my_gym.envs:SimplePuzzleHighResPixels',
    max_episode_steps=800,
    reward_threshold=500,
)