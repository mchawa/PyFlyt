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

from PyFlyt.gym_envs.quadx_mod_envs.hovering.quadx_hovering_env import QuadXHoverEnv
from PyFlyt.gym_envs.quadx_mod_envs.hovering.quadx_hovering_logger import Logger
from PyFlyt.gym_envs.quadx_mod_envs.hovering.quadx_hovering_pid_expert import (
    HoveringPIDExpert,
)

# from stable_baselines3.common.vec_env.VecMonitor import VecMonitor


project_dir = str(Path(__file__).resolve().parent.parent.parent)
if project_dir not in sys.path:
    sys.path.append(project_dir)

target_pos = np.array([5, 5, -5])
target_psi = np.deg2rad(90)

model = HoveringPIDExpert(taget_pos=target_pos, target_psi=target_psi)

log_dir = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/hovering/pid_results"

# Evaluate the agent
mean_reward_list = []
std_reward_list = []

eval_env_kwargs = {}
eval_env_kwargs["control_hz"] = 80
eval_env_kwargs["orn_conv"] = "NED_FRD"
eval_env_kwargs["randomize_start"] = False
eval_env_kwargs["target_pos"] = target_pos
eval_env_kwargs["target_psi"] = target_psi
eval_env_kwargs["start_pos"] = np.array([[4, 6, -4]])
eval_env_kwargs["start_orn"] = np.array([np.deg2rad([-10, 10, -90])])
eval_env_kwargs["min_pwm"] = 0.0
eval_env_kwargs["max_pwm"] = 1.0
eval_env_kwargs["noisy_motors"] = True
eval_env_kwargs["drone_model"] = "cf2x"
eval_env_kwargs["flight_mode"] = 7
eval_env_kwargs["simulate_wind"] = True
eval_env_kwargs["base_wind_velocities"] = np.array([4.0, -4.0, -1])
# eval_env_kwargs["base_wind_velocities"] = None
eval_env_kwargs["max_gust_strength"] = 7.0
# eval_env_kwargs["max_gust_strength"] = None
eval_env_kwargs["flight_dome_size"] = 100
eval_env_kwargs["max_duration_seconds"] = 10
eval_env_kwargs["angle_representation"] = "euler"
eval_env_kwargs["hovering_dome_size"] = 10.0
eval_env_kwargs["normalize_actions"] = False
eval_env_kwargs["normalize_obs"] = False
eval_env_kwargs["alpha"] = 2
eval_env_kwargs["beta"] = 0.1
eval_env_kwargs["gamma"] = 8
eval_env_kwargs["delta"] = 0.1
# eval_env_kwargs["render_mode"] = "human"
eval_env_kwargs["render_mode"] = None
# eval_env_kwargs["logger"] = None
eval_env_kwargs["logger"] = Logger(log_dir=log_dir)

eval_env = QuadXHoverEnv(**eval_env_kwargs)

eval_env = Monitor(eval_env)

mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    deterministic=True,
    render=(eval_env_kwargs["render_mode"] != None),
    return_episode_rewards=True,
    n_eval_episodes=1,
)

print("Mead Reward: {}, Std Reward: {}".format(mean_reward, std_reward))
