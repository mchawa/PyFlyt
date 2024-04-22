import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from PyFlyt.gym_envs.quadx_mod_envs.quadx_hovering_env import QuadXHoverEnv
from PyFlyt.rl_training.custom_eval_callback import CustomEvalCallback
from PyFlyt.rl_training.custom_feature_extractor import CustomFeatureExtractor

project_dir = str(Path(__file__).resolve().parent.parent.parent)
if project_dir not in sys.path:
    sys.path.append(project_dir)

if __name__ == "__main__":

    np.seterr(all="raise")
    th.autograd.set_detect_anomaly(True)
    set_random_seed(0)

    parser = argparse.ArgumentParser()

    # Environment Args
    parser.add_argument("--orn_conv", type=str, default="NED_FRD")
    parser.add_argument("--randomize_start", type=bool, default=True)
    parser.add_argument("--start_pos", type=float, nargs="+", default=[0.0, 0.0, -1.0])
    parser.add_argument("--start_orn", type=float, nargs="+", default=[0.0, 0.0, 0.0])
    parser.add_argument("--min_pwm", type=float, default=0.0)
    parser.add_argument("--max_pwm", type=float, default=1.0)
    parser.add_argument("--noisy_motors", type=bool, default=False)
    parser.add_argument("--drone_model", type=str, default="cf2x")
    parser.add_argument("--flight_mode", type=int, default=9)
    parser.add_argument("--simulate_wind", type=bool, default=False)
    parser.add_argument("--flight_dome_size", type=float, default=100)
    parser.add_argument("--max_duration_seconds", type=float, default=10.0)
    parser.add_argument("--angle_representation", type=str, default="euler")
    parser.add_argument("--hovering_dome_size", type=float, default=10.0)
    parser.add_argument("--normalize_actions", type=bool, default=True)
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--delta", type=float, default=0.1)

    # Training Args
    # parser.add_argument("--num_of_steps", type=int, default=8640000)
    parser.add_argument("--num_of_steps", type=int, default=50000000)
    parser.add_argument("--update_each_steps", type=int, default=3840)
    parser.add_argument("--batch_size", type=int, default=120)
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--num_of_layers", type=int, default=2)
    parser.add_argument("--layer_size", type=int, default=256)
    parser.add_argument("--eval_freq_multiplier", type=int, default=4)
    # parser.add_argument("--num_of_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--num_of_workers", type=int, default=1)

    args = parser.parse_args()

    start_time = datetime.datetime.now()
    output_dir_name = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    output_save_path = os.path.join(
        project_dir,
        "rl_training",
        "hovering",
        "trained_models",
        output_dir_name,
    )
    os.makedirs(output_save_path, exist_ok=True)

    tensorboard_log_path = os.path.join(output_save_path, "tensorboard")

    info_save_path = os.path.join(output_save_path, "info.txt")

    with open(info_save_path, "w+") as f:
        f.write("Arguments: {}\n".format(json.dumps(args.__dict__, indent=4)))
        f.write("Start Time: {}\n".format(start_time))

    # net_arch = [args.layer_size for _ in range(args.num_of_layers)]
    # net_arch = dict(vf=[256, 128, 64, 32], pi=[256, 128, 64, 32])
    # net_arch.append({"vf": [128], "pi": [64]})

    # policy_kwargs = {
    #     "net_arch": net_arch,
    #     "features_extractor_class": CustomFeatureExtractor,
    #     "features_extractor_kwargs": {"features_dim": 256},
    #     "share_features_extractor": True,
    # }

    # Create hovering environment
    env_kwargs = {}
    env_kwargs["orn_conv"] = args.orn_conv
    env_kwargs["randomize_start"] = args.randomize_start
    env_kwargs["start_pos"] = np.array([args.start_pos])
    env_kwargs["start_orn"] = np.array([args.start_orn])
    env_kwargs["min_pwm"] = args.min_pwm
    env_kwargs["max_pwm"] = args.max_pwm
    env_kwargs["noisy_motors"] = args.noisy_motors
    env_kwargs["drone_model"] = args.drone_model
    env_kwargs["flight_mode"] = args.flight_mode
    env_kwargs["simulate_wind"] = args.simulate_wind
    env_kwargs["flight_dome_size"] = args.flight_dome_size
    env_kwargs["max_duration_seconds"] = args.max_duration_seconds
    env_kwargs["angle_representation"] = args.angle_representation
    env_kwargs["hovering_dome_size"] = args.hovering_dome_size
    env_kwargs["normalize_actions"] = args.normalize_actions
    env_kwargs["alpha"] = args.alpha
    env_kwargs["beta"] = args.beta
    env_kwargs["gamma"] = args.gamma
    env_kwargs["delta"] = args.delta
    # env_kwargs["render_mode"] = None
    env_kwargs["render_mode"] = "human"

    env = make_vec_env(
        env_id=QuadXHoverEnv,
        n_envs=args.num_of_workers,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs["render_mode"] = None
    # eval_env_kwargs["render_mode"] = "human"

    eval_env = make_vec_env(
        env_id=QuadXHoverEnv,
        n_envs=1,
        env_kwargs=eval_env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=5,
        eval_freq=(args.eval_freq_multiplier * (args.update_each_steps) + 1),
        log_path=output_save_path,
        best_model_save_path=output_save_path,
        render=False,
        deterministic=True,
    )

    # model = PPO.load(
    #     path="/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/trained_models/PPO/2024_03_20_03_58_02/best_model_27_0_36.zip",
    #     env=env,
    #     tensorboard_log=tensorboard_log_path,
    #     print_system_info=True,
    #     verbose=1,
    # )

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=args.batch_size,
        n_steps=args.update_each_steps,
        n_epochs=args.n_epochs,
        tensorboard_log=tensorboard_log_path,
        # policy_kwargs=policy_kwargs,
        verbose=1,
    )

    model.learn(total_timesteps=args.num_of_steps, callback=eval_callback)

    end_time = datetime.datetime.now()

    with open(info_save_path, "w+") as f:
        f.write("End Time: {}\n".format(end_time))
        f.write("Total Time: {}\n".format(end_time - start_time))
