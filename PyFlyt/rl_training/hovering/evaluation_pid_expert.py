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

from PyFlyt.gym_envs.quadx_mod_envs.hovering.quadx_hovering_pid_expert import (
    HoveringPIDExpert,
)
from PyFlyt.gym_envs.quadx_mod_envs.quadx_hovering_env import QuadXHoverEnv
from PyFlyt.gym_envs.quadx_mod_envs.quadx_hovering_logger import Logger

# from stable_baselines3.common.vec_env.VecMonitor import VecMonitor


project_dir = str(Path(__file__).resolve().parent.parent.parent)
if project_dir not in sys.path:
    sys.path.append(project_dir)

model = HoveringPIDExpert(taget_pos=np.array([0, 0, -5]), target_psi=0)

log_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/hovering/pid_log.csv"

# Evaluate the agent
mean_reward_list = []
std_reward_list = []

eval_env_kwargs = {}
eval_env_kwargs["orn_conv"] = "NED_FRD"
eval_env_kwargs["randomize_start"] = False
eval_env_kwargs["start_pos"] = np.array([[0, 0, -5.0]])
eval_env_kwargs["start_orn"] = np.array([[-0.15708, -0.15708, 0]])
eval_env_kwargs["min_pwm"] = 0.0
eval_env_kwargs["max_pwm"] = 1.0
eval_env_kwargs["noisy_motors"] = False
eval_env_kwargs["drone_model"] = "cf2x"
eval_env_kwargs["flight_mode"] = 7
eval_env_kwargs["simulate_wind"] = False
eval_env_kwargs["flight_dome_size"] = 100
eval_env_kwargs["max_duration_seconds"] = 60 * 3
eval_env_kwargs["angle_representation"] = "euler"
eval_env_kwargs["hovering_dome_size"] = 10.0
eval_env_kwargs["normalize_actions"] = False
eval_env_kwargs["normalize_obs"] = True
eval_env_kwargs["alpha"] = 2
eval_env_kwargs["beta"] = 0.1
eval_env_kwargs["gamma"] = 2
eval_env_kwargs["delta"] = 0.1
eval_env_kwargs["render_mode"] = "human"
eval_env_kwargs["orn_conv"] = "NED_FRD"
eval_env_kwargs["logger"] = Logger(log_file_path=log_file_path)

eval_env = QuadXHoverEnv(**eval_env_kwargs)

eval_env = Monitor(eval_env)

mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    deterministic=True,
    render=(eval_env_kwargs["render_mode"] != None),
    n_eval_episodes=1,
)

print("Mead Reward: {}, Std Reward: {}".format(mean_reward, std_reward))

input("Press Enter to exit")