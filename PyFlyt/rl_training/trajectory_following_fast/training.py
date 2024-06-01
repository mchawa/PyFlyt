import argparse
import datetime
import json
import os
import signal
import socket
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

from PyFlyt.gym_envs.quadx_mod_envs.trajectory_following_fast.quadx_trajectory_following_env import (
    QuadXTrajectoryFollowingrEnv,
)
from PyFlyt.rl_training.custom_eval_callback import CustomEvalCallback
from PyFlyt.rl_training.custom_feature_extractor import CustomFeatureExtractor

project_dir = str(Path(__file__).resolve().parent.parent.parent)
if project_dir not in sys.path:
    sys.path.append(project_dir)

device_name = socket.gethostname()


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")

    if info_save_path is not None:
        end_time = datetime.datetime.now()

        with open(info_save_path, "a+") as f:
            f.write("End Time: {}\n".format(end_time))
            f.write("Total Time: {}\n".format(end_time - start_time))

    sys.exit(0)  # Exits the program cleanly


if __name__ == "__main__":

    np.seterr(all="raise")
    th.autograd.set_detect_anomaly(True)
    set_random_seed(0)

    parser = argparse.ArgumentParser()

    # Environment Args
    parser.add_argument("--control_hz", type=int, default=80)
    parser.add_argument("--orn_conv", type=str, default="NED_FRD")
    parser.add_argument("--randomize_start", type=bool, default=True)
    parser.add_argument("--start_pos", type=float, nargs="+", default=[0.0, 0.0, -5.0])
    parser.add_argument("--start_orn", type=float, nargs="+", default=[0.0, 0.0, 0.0])
    parser.add_argument("--random_trajectory", type=bool, default=True)
    parser.add_argument(
        "--waypoints",
        type=float,
        nargs="+",
        default=[[5.0, 5.0, -7.0], [5.0, -5.0, -7.0], [5.0, 5.0, -7.0]],
    )
    parser.add_argument("--goal_reach_distance", type=float, default=1)
    parser.add_argument("--min_pwm", type=float, default=0.0)
    parser.add_argument("--max_pwm", type=float, default=1.0)
    parser.add_argument("--noisy_motors", type=bool, default=True)
    parser.add_argument("--drone_model", type=str, default="cf2x")
    parser.add_argument("--flight_mode", type=int, default=8)
    parser.add_argument("--simulate_wind", type=bool, default=True)
    parser.add_argument("--flight_dome_size", type=float, default=100)
    parser.add_argument("--max_duration_seconds", type=float, default=30.0)
    parser.add_argument("--angle_representation", type=str, default="euler")
    parser.add_argument("--normalize_obs", type=bool, default=True)
    parser.add_argument("--normalize_actions", type=bool, default=True)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--gamma", type=float, default=0.2)

    # Training Args
    parser.add_argument("--num_of_layers", type=int, default=2)
    parser.add_argument("--layer_size", type=int, default=256)
    parser.add_argument("--num_of_workers", type=int, default=mp.cpu_count())
    # parser.add_argument("--num_of_workers", type=int, default=1)
    parser.add_argument("--eval_freq_multiplier", type=int, default=4)
    batch_size = parser.get_default("control_hz") * 1
    parser.add_argument("--batch_size", type=int, default=batch_size)
    update_each_steps = batch_size * 32
    parser.add_argument("--update_each_steps", type=int, default=update_each_steps)
    parser.add_argument("--n_epochs", type=int, default=15)
    num_of_steps = (
        np.ceil(
            100000000
            / (
                update_each_steps
                * parser.get_default("num_of_workers")
                * parser.get_default("eval_freq_multiplier")
            )
        )
        * (
            update_each_steps
            * parser.get_default("num_of_workers")
            * parser.get_default("eval_freq_multiplier")
        )
    ) + 1
    parser.add_argument("--num_of_steps", type=int, default=num_of_steps)

    args = parser.parse_args()

    # net_arch = [args.layer_size for _ in range(args.num_of_layers)]
    net_arch = dict(pi=[64, 64, 32, 32], vf=[64, 64, 32, 32])
    # net_arch.append({"vf": [128], "pi": [64]})

    policy_kwargs = {
        "net_arch": net_arch,
        # "features_extractor_class": CustomFeatureExtractor,
        # "features_extractor_kwargs": {"features_dim": 256},
        # "share_features_extractor": True,
    }

    start_time = datetime.datetime.now()
    output_dir_name = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    output_save_path = os.path.join(
        project_dir,
        "rl_training",
        "trajectory_following_fast",
        "trained_models",
        output_dir_name,
    )
    os.makedirs(output_save_path, exist_ok=True)

    tensorboard_log_path = os.path.join(output_save_path, "tensorboard")

    info_save_path = os.path.join(output_save_path, "info.txt")

    with open(info_save_path, "w+") as f:
        f.write("Device Name: {}\n".format(device_name))
        f.write("Arguments: {}\n".format(json.dumps(args.__dict__, indent=4)))
        f.write("Policy Arguments: {}\n".format(json.dumps(policy_kwargs, indent=4)))
        f.write("Start Time: {}\n".format(start_time))

    signal.signal(signal.SIGINT, signal_handler)  # Register the signal handler

    # Create trajectory following environment
    env_kwargs = {}
    env_kwargs["control_hz"] = args.control_hz
    env_kwargs["orn_conv"] = args.orn_conv
    env_kwargs["randomize_start"] = args.randomize_start
    env_kwargs["start_pos"] = np.array([args.start_pos])
    env_kwargs["start_orn"] = np.array([args.start_orn])
    env_kwargs["random_trajectory"] = args.random_trajectory
    env_kwargs["waypoints"] = np.array(args.waypoints)
    env_kwargs["min_pwm"] = args.min_pwm
    env_kwargs["max_pwm"] = args.max_pwm
    env_kwargs["noisy_motors"] = args.noisy_motors
    env_kwargs["drone_model"] = args.drone_model
    env_kwargs["flight_mode"] = args.flight_mode
    env_kwargs["simulate_wind"] = args.simulate_wind
    env_kwargs["flight_dome_size"] = args.flight_dome_size
    env_kwargs["max_duration_seconds"] = args.max_duration_seconds
    env_kwargs["angle_representation"] = args.angle_representation
    env_kwargs["normalize_actions"] = args.normalize_actions
    env_kwargs["normalize_obs"] = args.normalize_obs
    env_kwargs["alpha"] = args.alpha
    env_kwargs["beta"] = args.beta
    env_kwargs["gamma"] = args.gamma
    env_kwargs["draw_waypoints"] = False
    env_kwargs["render_mode"] = None
    env_kwargs["logger"] = None

    env = make_vec_env(
        env_id=QuadXTrajectoryFollowingrEnv,
        n_envs=args.num_of_workers,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs["draw_waypoints"] = False
    eval_env_kwargs["render_mode"] = None
    # eval_env_kwargs["render_mode"] = "human"

    eval_env = make_vec_env(
        env_id=QuadXTrajectoryFollowingrEnv,
        n_envs=args.num_of_workers,
        env_kwargs=eval_env_kwargs,
        vec_env_cls=SubprocVecEnv,
    )

    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=10,
        eval_freq=(args.eval_freq_multiplier * (args.update_each_steps) + 1),
        log_path=output_save_path,
        best_model_save_path=output_save_path,
        render=(eval_env_kwargs["render_mode"] == "human"),
        deterministic=True,
    )

    model = PPO.load(
        path="/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/trajectory_following_fast/trained_models/2024_05_26_17_38_28/best_model_8_2401_0_35789_2973.zip",
        env=env,
        tensorboard_log=tensorboard_log_path,
        print_system_info=True,
        verbose=1,
    )

    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     batch_size=args.batch_size,
    #     n_steps=args.update_each_steps,
    #     n_epochs=args.n_epochs,
    #     tensorboard_log=tensorboard_log_path,
    #     policy_kwargs=policy_kwargs,
    #     verbose=1,
    # )

    model.learn(total_timesteps=args.num_of_steps, callback=eval_callback)

    end_time = datetime.datetime.now()

    with open(info_save_path, "a+") as f:
        f.write("End Time: {}\n".format(end_time))
        f.write("Total Time: {}\n".format(end_time - start_time))
