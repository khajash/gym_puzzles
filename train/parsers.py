import argparse


def pop_arguments(config, keys):
    """Split arguments from config dict

    Args:
        config (dict_): Dictionary to pop keys from
        keys (list of strings): List of keys to pop from config

    Returns:
        (dict, dict): dictionary minus keys, dictionary of keys
    """
    alg_config = {}
    for hparam in keys:
        if hparam in config:
            val = config.pop(hparam)
            alg_config[hparam] = val
    return config, alg_config


def setup_base_parser():
    
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--config",
        default="./configs/ppo-mrp-v3.json", 
        type=str,
        help="Config json filename.",
    )
    parser.add_argument(
        "--seed",
        default=17,
        type=int,
        help="Random seed. (int, default = 17)",
    )
    parser.add_argument(
        "--total_timesteps",
        default=1_000_000,
        type=int,
        help="Number of timesteps to run for training stable baselines. (int, default = 1M)",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Do not initialize wandb logger"
    )
    # parser.add_argument(
    #     "--new_step_api",
    #     action="store_true",
    #     help="Use new step API in OpenAI env (new_obs, rew, term, trunc, info) vs (new_obs, rew, done, info)"
    # )
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
    parser.add_argument(
        "--run_wandb_sweep",
        action="store_true",
        help="Run wandb sweep and use alg parser to overwrite config.",
    )
    parser.add_argument(
        "--n_envs",
        default=1,
        type=int,
    )

    return parser


def setup_PPO_parser(parent_parser):
    """Setup parser including PPO hparams. Use this when using sweep from wandb, otherwise
    this will override the config file settings.

    Args:
        parent_parser (ArgumentParser): Parent parser to add onto.

    Returns:
        ArgumentParser: parser with PPO arguments added
    """
    
    parser = argparse.ArgumentParser(parents=[parent_parser])
    parser.add_argument(
        "--learning_rate",
        default=0.0003, 
        type=float,
    )
    parser.add_argument(
        "--clip_range",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--n_epochs",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--ent_coef",
        default=0.01,
        type=float
    )
    parser.add_argument(
        "--n_steps",
        default=4096,
        type=int,
    )
    parser.add_argument(
        "--max_grad_norm",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--vf_coef",
        default=0.5,
        type=float
    )

    return parser


PPO_HPARAMS = [
        'learning_rate',
        'clip_range',
        'batch_size',
        'n_epochs',
        'ent_coef',
        'n_steps',
        'max_grad_norm',
        'vf_coef'
    ]