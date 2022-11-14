import json
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecVideoRecorder


import parsers
import gym
import gym_puzzles

def test():


    # Create environment
    env = gym.make("LunarLander-v2")

    # Instantiate the agent
    model = DQN("MlpPolicy", env, verbose=1)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5)) #, progress_bar=True)
    # Save the agent
    model.save("dqn_lunar")
    del model  # delete trained model to demonstrate loading

    # Load the trained agent
    # NOTE: if you have loading issue, you can pass `print_system_info=True`
    # to compare the system on which the model was trained vs the current one
    # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
    model = DQN.load("dqn_lunar", env=env)

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

def main():

    parser = parsers.setup_base_parser()
    cl_config = vars(parser.parse_args())
    training_run = '25z46j4s'

    # Load config file
    with open(cl_config['config'], "r") as f:
        config = json.load(f)

    config.update(**cl_config)

    seed_base = config['seed']

    print(config)    

    env = gym.make(config['env'])
    env = DummyVecEnv([lambda: gym.make(config['env'])])
    env = VecVideoRecorder(
        env, f"./models/{training_run}", 
        record_video_trigger=lambda x: x == 0, video_length=300)

    env = VecNormalize.load(f"./models/{training_run}/saved_env.pkl", env)
    env.training = False
    env.norm_reward = False

    # TODO: issue loading env trained with multiple envs and potentially vecNormalize - look into
    # khajash/MultiRobotPuzzle-v3/nawfz2ds
    # khajash/MultiRobotPuzzle-v3/25z46j4s
    # khajash/MultiRobotPuzzle-v3/3k9r89ze
    model = PPO.load(f"./models/{training_run}/ppo_puzzles_v2", env=env, print_system_info=True)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"mean reward: {mean_reward}, std reward: {std_reward}")
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
    # test()