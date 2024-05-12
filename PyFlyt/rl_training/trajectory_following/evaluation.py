import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from PyFlyt.gym_envs.quadx_mod_envs.trajectory_following.quadx_trajectory_following_env import (
    QuadXTrajectoryFollowingrEnv,
)
from PyFlyt.gym_envs.quadx_mod_envs.trajectory_following.quadx_trajectory_following_logger import (
    Logger,
)

project_dir = str(Path(__file__).resolve().parent.parent.parent)
if project_dir not in sys.path:
    sys.path.append(project_dir)

model_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/trajectory_following/trained_models/2024_05_12_02_57_30/best_model_16_801_0_6344_503.zip"

log_file_path = model_path.replace(".zip", ".csv")

model = PPO.load(model_path, print_system_info=True)

# policy_parameters = model.get_parameters()["policy"]

# for key in policy_parameters:
#     policy_parameters[key] = policy_parameters[key].cpu().detach().numpy().tolist()

# with open("data.json", "w", encoding="utf-8") as f:
#     json.dump(policy_parameters, f, ensure_ascii=False, indent=4)

# Evaluate the agent
waypoints = np.array(
    [
        [4.05, 2.94, -6.0],
        [1.55, 4.76, -7.0],
        [-1.55, 4.76, -8.0],
        [-4.05, 2.94, -9.0],
        [-5.0, 0.0, -10.0],
        [-4.05, -2.94, -9.0],
        [-1.55, -4.76, -8.0],
        [1.55, -4.76, -7.0],
        [4.05, -2.94, -6.0],
        [5.0, 0.0, -5.0],
    ]
)

mean_reward_list = []
std_reward_list = []

eval_env_kwargs = {}
eval_env_kwargs["control_hz"] = 80
eval_env_kwargs["orn_conv"] = "NED_FRD"
eval_env_kwargs["randomize_start"] = False
eval_env_kwargs["start_pos"] = np.array([[5, 0, -5]])
eval_env_kwargs["start_orn"] = np.array([np.deg2rad([0, 0, 0])])
eval_env_kwargs["random_trajectory"] = False
eval_env_kwargs["waypoints"] = waypoints
eval_env_kwargs["maximum_velocity"] = 20.0
eval_env_kwargs["min_pwm"] = 0.0
eval_env_kwargs["max_pwm"] = 1.0
eval_env_kwargs["noisy_motors"] = True
eval_env_kwargs["drone_model"] = "cf2x"
eval_env_kwargs["flight_mode"] = 8
eval_env_kwargs["simulate_wind"] = False
eval_env_kwargs["flight_dome_size"] = 100
eval_env_kwargs["max_duration_seconds"] = 10
eval_env_kwargs["angle_representation"] = "euler"
eval_env_kwargs["normalize_actions"] = True
eval_env_kwargs["normalize_obs"] = True
eval_env_kwargs["alpha"] = 1
eval_env_kwargs["beta"] = 1
eval_env_kwargs["gamma"] = 0.2
eval_env_kwargs["delta"] = 1
eval_env_kwargs["draw_waypoints"] = True
eval_env_kwargs["render_mode"] = "human"
# eval_env_kwargs["render_mode"] = None
eval_env_kwargs["logger"] = Logger(log_file_path=log_file_path)
# eval_env_kwargs["logger"] = None

eval_env = QuadXTrajectoryFollowingrEnv(**eval_env_kwargs)

eval_env = Monitor(eval_env)

ep_rewards, ep_lengths = evaluate_policy(
    model,
    eval_env,
    deterministic=True,
    render=(eval_env_kwargs["render_mode"] != None),
    n_eval_episodes=1,
    return_episode_rewards=True,
)

mean_reward = np.mean(ep_rewards)
std_reward = np.std(ep_rewards)

mean_length = np.mean(ep_lengths)
std_length = np.std(ep_lengths)

print("Evaluation Results:")
print("Ep Rewards: {}, Ep Lengths: {}".format(ep_rewards, ep_lengths))
print("Mead Reward: {}, Std Reward: {}".format(mean_reward, std_reward))
print("Mean Length: {}, Std Length: {}".format(mean_length, std_length))
