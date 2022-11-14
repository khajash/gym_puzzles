import numpy as np
import random
import gym
import gym_puzzles


def main():

    to_render = True
    seed = 17

    env = gym.make('MultiRobotPuzzle-v3', heavy=True)
    # env = gym.make('LunarLander-v2')
    
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    obs = env.reset()
    done = False

    print("obs shape", obs.shape, obs.min(), obs.max())
    
    for i in range(5):
        env.reset()
        while not done:
            action = env.action_space.sample()

            new_obs, rew, done, info = env.step(action)

            # print("done:", done)
            print(f"act: {action} rew: {rew}")
            # print("obs here: ", new_obs.shape, new_obs.min(), new_obs.max())
            if to_render:
                env.render(mode="human")



    env.close()

if __name__ == "__main__":
    main()


