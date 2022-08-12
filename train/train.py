import json

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback
import torch

import gym
import gym_puzzles


def setup_training_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/ppo-mrp-v0.json", 
        type=str,
        help="Save model when done.",
    )
    parser.add_argument(
        "--seed",
        default=17,
        type=int,
        help="Random seed. (int, default = 17)",
    )
    parser.add_argument(
        "--total_timesteps",
        default=1000000,
        type=int,
        help="Number of epochs to run the training. (int, default = 75)",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Do not initialize wandb logger"
    )
    parser.add_argument(
        "--new_step_api",
        action="store_true",
        help="Use new step API in OpenAI env (new_obs, rew, term, trunc, info) vs (new_obs, rew, done, info)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render env in testing"
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save model when done.",
    )

    return parser.parse_args()


def main():

    args = setup_training_parser()
    cl_config = vars(args)

    # Load config file
    with open(args.config, "r") as f:
        config = json.load(f)

    config.update(**cl_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running model on device {device}")

    use_wandb = not args.disable_wandb

    if use_wandb:
        run = wandb.init(
            project=config['env'], 
            group=f"{config['network']}-sb3-v0", 
            config=config, 
            sync_tensorboard=True, 
            monitor_gym=False, 
            save_code=True
        )

    print("\nConfig:\n", config)
    seed_base = config['seed']

    # env = gym.make(config['env'], **config['env_params'])
    
    def seed_make_env(seed):
        def make_env():
            env = gym.make(config['env'])
            env.seed(seed)
            env.action_space.seed(seed)
            env = Monitor(env)
            return env
        return make_env

    env_fn_list = []
    for i in range(config['n_envs']):
        s = seed_base + i*22
        env_fn_list.append(seed_make_env(s))

    print(f"Running {len(env_fn_list)} envs")

    # TODO: test out SubprocVecEnv
    env = DummyVecEnv(env_fn_list)
    # env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    env = VecNormalize(env)
    # print(config['alg_params'])
    
    model = PPO(
        env=env, 
        verbose=1, 
        tensorboard_log=f"runs/{run.id}",
        seed=config['seed'],
        device=device,
        **config['alg_params'],
    )
    
    model.learn(
        total_timesteps=config["total_timesteps"],
        log_interval=4,
        callback=WandbCallback(
            gradient_save_freq=100,
            log='all',
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    # model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    # env = make_env()
    # alg_config = config['alg_params']
    # model = DQN("MlpPolicy", env, verbose=1, 
    #     learning_rate=config['optim_params']['lr'], 
    #     buffer_size=int(alg_config['mem_size']),
    #     learning_starts=alg_config['learning_starts'],
    #     batch_size=alg_config['batch_size'],
    #     target_update_interval=alg_config['target_update'],
    #     gamma=alg_config['gamma'],
    #     tensorboard_log=f"runs/{run.id}",
    #     exploration_fraction=0.12, 
    #     exploration_final_eps=0.1,
    #     gradient_steps=-1,
    #     train_freq=1,
    #     policy_kwargs=dict(net_arch=[256, 256]),
    #     seed=config['seed']
    # )

    # # look at off policy algorithms learn
    # # separate learn and train function
    # model.learn(
    #     total_timesteps=100000, 
    #     log_interval=4, 
    #     callback=WandbCallback(
    #         gradient_save_freq=20,
    #         model_save_path=f"models/{run.id}", 
    #         log='all',
    #         verbose=2,
    #     ),
    # )
    model.save("ppo_puzzles_v2")

    del model # remove to demonstrate saving and loading


    model = PPO.load("ppo_puzzles_v2")

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