import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from PyFlyt.gym_envs.quadx_mod_envs.trajectory_following_slow.quadx_trajectory_following_env import (
    QuadXTrajectoryFollowingrEnv,
)
from PyFlyt.gym_envs.quadx_mod_envs.trajectory_following_slow.quadx_trajectory_following_logger import (
    Logger,
)
from PyFlyt.gym_envs.quadx_mod_envs.trajectory_following_slow.quadx_trajectory_following_pid_expert import (
    TrajectoryFollowingPIDExpert,
)

project_dir = str(Path(__file__).resolve().parent.parent.parent)
if project_dir not in sys.path:
    sys.path.append(project_dir)

model = TrajectoryFollowingPIDExpert()

log_dir = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/trajectory_following_slow/pid_results"

# Evaluate the agent
start_pos_scenario_1 = np.array([[5, 0, -5]])
start_orn_scenario_1 = np.array([np.deg2rad([0, 0, 0])])
waypoints_scenario_1 = np.array(
    [
        [4.05, 2.94, -6.0, 0.0],
        [1.55, 4.76, -7.0, np.deg2rad(20)],
        [-1.55, 4.76, -8.0, np.deg2rad(40)],
        [-4.05, 2.94, -9.0, np.deg2rad(60)],
        [-5.0, 0.0, -10.0, np.deg2rad(80)],
        [-4.05, -2.94, -9.0, np.deg2rad(100)],
        [-1.55, -4.76, -8.0, np.deg2rad(120)],
        [1.55, -4.76, -7.0, np.deg2rad(140)],
        [4.05, -2.94, -6.0, np.deg2rad(160)],
        [5.0, 0.0, -5.0, np.deg2rad(175)],
    ]
)
base_wind_velocity_scenario_1 = np.array([-2.0, -2.0, 0.5])

start_pos_scenario_2 = np.array([[0, 0, -5]])
start_orn_scenario_2 = np.array([np.deg2rad([0, 0, 0])])
waypoints_scenario_2 = np.array(
    [
        [0.0, 5.0, -5.0, np.deg2rad(35)],
        [5.0, 5.0, -5.0, np.deg2rad(70)],
        [5.0, 0.0, -5.0, np.deg2rad(105)],
        [0.0, 0.0, -5.0, np.deg2rad(140)],
        [0.0, 0.0, -10.0, np.deg2rad(175)],
        [0.0, 5.0, -10.0, np.deg2rad(140)],
        [5.0, 5.0, -10.0, np.deg2rad(105)],
        [5.0, 0.0, -10.0, np.deg2rad(70)],
        [0.0, 0.0, -10.0, np.deg2rad(35)],
        [0.0, 0.0, -5.0, np.deg2rad(0)],
    ]
)
base_wind_velocity_scenario_2 = np.array([2.0, 2.0, -0.5])

start_pos_scenario_3 = np.array([[5, 5, -10]])
start_orn_scenario_3 = np.array([np.deg2rad([0, 0, 0])])
waypoints_scenario_3 = np.array(
    [
        [-5.0, -5.0, -10.0, np.deg2rad(25)],
        [5.0, 5.0, -10.0, np.deg2rad(50)],
        [-5.0, -5.0, -10.0, np.deg2rad(75)],
        [5.0, 5.0, -10.0, np.deg2rad(100)],
        [-5.0, -5.0, -10.0, np.deg2rad(125)],
        [5.0, 5.0, -10.0, np.deg2rad(150)],
        [-5.0, -5.0, -10.0, np.deg2rad(175)],
        [5.0, 5.0, -10.0, np.deg2rad(150)],
        [-5.0, -5.0, -10.0, np.deg2rad(125)],
        [5.0, 5.0, -10.0, np.deg2rad(100)],
    ]
)
base_wind_velocity_scenario_3 = np.array([0.0, 0.0, 0.0])

mean_reward_list = []
std_reward_list = []

eval_env_kwargs = {}
eval_env_kwargs["control_hz"] = 80
eval_env_kwargs["orn_conv"] = "NED_FRD"
eval_env_kwargs["randomize_start"] = False
eval_env_kwargs["start_pos"] = start_pos_scenario_3
eval_env_kwargs["start_orn"] = start_orn_scenario_3
eval_env_kwargs["random_trajectory"] = False
eval_env_kwargs["goal_reach_distance"] = 0.3
eval_env_kwargs["goal_reach_angle"] = np.deg2rad(5)
eval_env_kwargs["waypoints"] = waypoints_scenario_3
eval_env_kwargs["min_pwm"] = 0.0
eval_env_kwargs["max_pwm"] = 1.0
eval_env_kwargs["noisy_motors"] = True
eval_env_kwargs["drone_model"] = "cf2x"
eval_env_kwargs["flight_mode"] = 10
eval_env_kwargs["simulate_wind"] = True
eval_env_kwargs["base_wind_velocities"] = base_wind_velocity_scenario_3
eval_env_kwargs["max_gust_strength"] = 7.0
eval_env_kwargs["flight_dome_size"] = 100
eval_env_kwargs["max_duration_seconds"] = 30
eval_env_kwargs["angle_representation"] = "euler"
eval_env_kwargs["normalize_actions"] = False
eval_env_kwargs["normalize_obs"] = False
eval_env_kwargs["alpha"] = 2
eval_env_kwargs["beta"] = 4
eval_env_kwargs["gamma"] = 0.2
eval_env_kwargs["draw_waypoints"] = True
# eval_env_kwargs["render_mode"] = "human"
eval_env_kwargs["render_mode"] = None
eval_env_kwargs["logger"] = Logger(log_dir=log_dir)
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
