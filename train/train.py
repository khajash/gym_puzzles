import json

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback
import torch

import parsers as parse
import gym
import gym_puzzles


def main():

    # Set this to True if you want to override config params with commandline params
    RUN_WANDB_SWEEP = False

    # Setup argument parsers - go to parsers.py for more info or use --help
    parent_parser = parse.setup_base_parser()
    parser = parse.setup_PPO_parser(parent_parser)
    cl_config = vars(parser.parse_args())

    # Split alg params from base params
    cl_config, alg_config = parse.pop_arguments(cl_config, parse.PPO_HPARAMS)
    
    # print("\nalg config: ", alg_config)
    # print("\ncl config: ", cl_config)
    
    # Load config file
    with open(cl_config['config'], "r") as f:
        config = json.load(f)

    # Update base config with commandline params 
    config.update(**cl_config)

    # Overwrite config with commandline params 
    if RUN_WANDB_SWEEP:
        config['alg_params'].update(**alg_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running model on device {device}")

    use_wandb = not config['disable_wandb']

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
        # NOTE: none of these callbacks are working - FIX THIS
        callback=WandbCallback(
            gradient_save_freq=10,
            log='all',
            model_save_freq=10,
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



if __name__ == "__main__":
    main()
    # test()