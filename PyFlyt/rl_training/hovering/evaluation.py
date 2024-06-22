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

# from stable_baselines3.common.vec_env.VecMonitor import VecMonitor


project_dir = str(Path(__file__).resolve().parent.parent.parent)
if project_dir not in sys.path:
    sys.path.append(project_dir)

model_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/hovering/trained_models/2024_06_14_19_18_13/best_model_23_801_0_25835_723.zip"

log_dir = model_path.replace(".zip", "_results")

model = PPO.load(model_path, print_system_info=True)

# policy_parameters = model.get_parameters()["policy"]

# for key in policy_parameters:
#     policy_parameters[key] = policy_parameters[key].cpu().detach().numpy().tolist()

# with open("data.json", "w", encoding="utf-8") as f:
#     json.dump(policy_parameters, f, ensure_ascii=False, indent=4)

# Evaluate the agent
mean_reward_list = []
std_reward_list = []

eval_env_kwargs = {}
eval_env_kwargs["control_hz"] = 80
eval_env_kwargs["orn_conv"] = "NED_FRD"
eval_env_kwargs["randomize_start"] = False
eval_env_kwargs["target_pos"] = np.array([10, -10, -5])
eval_env_kwargs["target_psi"] = np.deg2rad(-90)
eval_env_kwargs["start_pos"] = np.array([[19, -19, -14]])
eval_env_kwargs["start_orn"] = np.array([np.deg2rad([-10, 10, 90])])
eval_env_kwargs["min_pwm"] = 0.0
eval_env_kwargs["max_pwm"] = 1.0
eval_env_kwargs["noisy_motors"] = True
eval_env_kwargs["drone_model"] = "cf2x"
eval_env_kwargs["flight_mode"] = 8
eval_env_kwargs["simulate_wind"] = True
eval_env_kwargs["base_wind_velocities"] = np.array([5.0, -5.0, -1.0])
# eval_env_kwargs["base_wind_velocities"] = None
eval_env_kwargs["max_gust_strength"] = 7.0
# eval_env_kwargs["max_gust_strength"] = None
eval_env_kwargs["flight_dome_size"] = 100
eval_env_kwargs["max_duration_seconds"] = 10
eval_env_kwargs["angle_representation"] = "euler"
eval_env_kwargs["normalize_actions"] = True
eval_env_kwargs["normalize_obs"] = True
eval_env_kwargs["alpha"] = 2
eval_env_kwargs["beta"] = 0.1
eval_env_kwargs["gamma"] = 4
eval_env_kwargs["delta"] = 0.1
# eval_env_kwargs["render_mode"] = "human"
eval_env_kwargs["render_mode"] = None
eval_env_kwargs["logger"] = Logger(log_dir=log_dir)
# eval_env_kwargs["logger"] = None
#
eval_env = QuadXHoverEnv(**eval_env_kwargs)

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
