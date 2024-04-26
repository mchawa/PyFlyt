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

from PyFlyt.gym_envs.quadx_mod_envs.quadx_hovering_env import QuadXHoverEnv

project_dir = str(Path(__file__).resolve().parent.parent.parent)
if project_dir not in sys.path:
    sys.path.append(project_dir)

model_path = os.path.join(
    project_dir,
    "rl_training",
    "hovering",
    "trained_models",
    # "Important_Models",
    "2024_04_25_01_14_18",
    "best_model_59_1201_0_22864_310.zip",
)

model = PPO.load(model_path, print_system_info=True)

policy_parameters = model.get_parameters()["policy"]

# for key in policy_parameters:
#     policy_parameters[key] = policy_parameters[key].cpu().detach().numpy().tolist()

# with open("data.json", "w", encoding="utf-8") as f:
#     json.dump(policy_parameters, f, ensure_ascii=False, indent=4)

# Evaluate the agent
mean_reward_list = []
std_reward_list = []

eval_env_kwargs = {}
eval_env_kwargs["orn_conv"] = "NED_FRD"
eval_env_kwargs["randomize_start"] = False
eval_env_kwargs["start_pos"] = np.array([[0, 0, -5.0]])
eval_env_kwargs["start_orn"] = np.array([[0.15708, -0.15708, 0]])
eval_env_kwargs["min_pwm"] = 0.0
eval_env_kwargs["max_pwm"] = 1.0
eval_env_kwargs["noisy_motors"] = True
eval_env_kwargs["drone_model"] = "cf2x"
eval_env_kwargs["flight_mode"] = 8
eval_env_kwargs["simulate_wind"] = False
eval_env_kwargs["flight_dome_size"] = 100
eval_env_kwargs["max_duration_seconds"] = 10
eval_env_kwargs["angle_representation"] = "euler"
eval_env_kwargs["hovering_dome_size"] = 10.0
eval_env_kwargs["normalize_actions"] = True
eval_env_kwargs["normalize_obs"] = True
eval_env_kwargs["alpha"] = 2
eval_env_kwargs["beta"] = 0.1
eval_env_kwargs["gamma"] = 2
eval_env_kwargs["delta"] = 0.1
eval_env_kwargs["render_mode"] = "human"
eval_env_kwargs["orn_conv"] = "NED_FRD"

eval_env = QuadXHoverEnv(**eval_env_kwargs)

mean_reward, std_reward = evaluate_policy(
    model, eval_env, deterministic=True, render=True, n_eval_episodes=10
)

print("Mead Reward: {}, Std Reward: {}".format(mean_reward, std_reward))

input("Press Enter to exit")
