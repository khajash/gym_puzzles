import json
from stable_baselines3 import PPO

import parsers
import gym
import gym_puzzles


def main():

    parser = parsers.setup_base_parser()
    cl_config = vars(parser.parse_args())

    # Load config file
    with open(cl_config['config'], "r") as f:
        config = json.load(f)

    config.update(**cl_config)

    seed_base = config['seed']

    print(config)    

    env = gym.make(config['env'])

    # TODO: issue loading env trained with multiple envs and potentially vecNormalize - look into
    model = PPO.load("ppo_puzzles_v2.1")

    obs = env.reset()
    for i in range(5):
        obs = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                break
    env.close()

if __name__ == "__main__":
    main()