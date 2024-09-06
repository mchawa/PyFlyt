import os

import matplotlib.pyplot as plt
import numpy as np


def load_csv(file_path):
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    return data


def log_episode(scenario_name, data1, data2, data3):
    """
    Write episode log into specified file

        Arguments:
            path_to_log(str): path to directory where to write info
            name(str): name for accumulated log info

    """
    fig1, ax1 = plt.subplots(3, 3, layout="constrained")

    # Plot the target position and the actual position against time
    time = np.array([entry[0] for entry in data1]) * 1 / 80
    target_pos = np.array([entry[1:4] for entry in data1])
    actual_pos_rl = np.array([entry[6:9] for entry in data1])
    actual_pos_cascaded_pid = np.array([entry[6:9] for entry in data2])
    actual_pos_gain_scheduling_pid = np.array([entry[6:9] for entry in data3])
    ax1[0, 0].plot(time, target_pos[:, 0], label="Reference")
    ax1[0, 0].plot(time, actual_pos_rl[:, 0], label="RL")
    ax1[0, 0].plot(time, actual_pos_cascaded_pid[:, 0], label="Cascaded PID")
    ax1[0, 0].plot(
        time, actual_pos_gain_scheduling_pid[:, 0], label="Gain-Scheduling PID"
    )
    ax1[0, 0].set_xlabel("Time (s)")
    ax1[0, 0].set_title("Linear Position X (m)")
    ax1[0, 0].legend()
    ax1[0, 1].plot(time, target_pos[:, 1], label="Reference")
    ax1[0, 1].plot(time, actual_pos_rl[:, 1], label="RL")
    ax1[0, 1].plot(time, actual_pos_cascaded_pid[:, 1], label="Cascaded PID")
    ax1[0, 1].plot(
        time, actual_pos_gain_scheduling_pid[:, 1], label="Gain-Scheduling PID"
    )
    ax1[0, 1].set_xlabel("Time (s)")
    ax1[0, 1].set_title("Linear Position Y (m)")
    ax1[0, 1].legend()
    ax1[0, 2].plot(time, target_pos[:, 2], label="Refernce")
    ax1[0, 2].plot(time, actual_pos_rl[:, 2], label="RL")
    ax1[0, 2].plot(time, actual_pos_cascaded_pid[:, 2], label="Cascaded PID")
    ax1[0, 2].plot(
        time, actual_pos_gain_scheduling_pid[:, 2], label="Gain-Scheduling PID"
    )
    ax1[0, 2].set_xlabel("Time (s)")
    ax1[0, 2].set_title("Linear Position Z (m)")
    ax1[0, 2].legend()

    # Plot the target orientation and the actual orientation in rad against time
    actual_phi_rad_rl = np.array([entry[12] for entry in data1])
    actual_phi_rad_cascaded_pid = np.array([entry[12] for entry in data2])
    actual_phi_rad_gain_scheduling_pid = np.array([entry[12] for entry in data3])
    actual_theta_rad_rl = np.array([entry[14] for entry in data1])
    actual_theta_rad_cascaded_pid = np.array([entry[14] for entry in data2])
    actual_theta_rad_gain_scheduling_pid = np.array([entry[14] for entry in data3])
    actual_psi_rad_rl = np.array([entry[16] for entry in data1])
    actual_psi_rad_cascaded_pid = np.array([entry[16] for entry in data2])
    actual_psi_rad_gain_scheduling_pid = np.array([entry[16] for entry in data3])
    target_psi_rad_rl = np.array([entry[4] for entry in data1])
    ax1[1, 0].plot(time, actual_phi_rad_rl, label="RL")
    ax1[1, 0].plot(time, actual_phi_rad_cascaded_pid, label="Cascaded PID")
    ax1[1, 0].plot(
        time, actual_phi_rad_gain_scheduling_pid, label="Gain-Scheduling PID"
    )
    ax1[1, 0].set_xlabel("Time (s)")
    ax1[1, 0].set_title("Roll Angle (rad)")
    ax1[1, 0].legend()
    ax1[1, 1].plot(time, actual_theta_rad_rl, label="RL")
    ax1[1, 1].plot(time, actual_theta_rad_cascaded_pid, label="Cascaded PID")
    ax1[1, 1].plot(
        time, actual_theta_rad_gain_scheduling_pid, label="Gain-Scheduling PID"
    )
    ax1[1, 1].set_xlabel("Time (s)")
    ax1[1, 1].set_title("Pitch Angle (rad)")
    ax1[1, 1].legend()
    ax1[1, 2].plot(time, target_psi_rad_rl, label="Reference")
    ax1[1, 2].plot(time, actual_psi_rad_rl, label="RL")
    ax1[1, 2].plot(time, actual_psi_rad_cascaded_pid, label="Cascaded PID")
    ax1[1, 2].plot(
        time, actual_psi_rad_gain_scheduling_pid, label="Gain-Scheduling PID"
    )
    ax1[1, 2].set_xlabel("Time (s)")
    ax1[1, 2].set_title("Yaw Angle (rad)")
    ax1[1, 2].legend()

    # Plot the target orientation and the actual orientation in deg against time
    actual_phi_deg_rl = np.array([entry[13] for entry in data1])
    actual_phi_deg_cascaded_pid = np.array([entry[13] for entry in data2])
    actual_phi_deg_gain_scheduling_pid = np.array([entry[13] for entry in data3])
    actual_theta_deg_rl = np.array([entry[15] for entry in data1])
    actual_theta_deg_cascaded_pid = np.array([entry[15] for entry in data2])
    actual_theta_deg_gain_scheduling_pid = np.array([entry[15] for entry in data3])
    actual_psi_deg_rl = np.array([entry[17] for entry in data1])
    actual_psi_deg_cascaded_pid = np.array([entry[17] for entry in data2])
    actual_psi_deg_gain_scheduling_pid = np.array([entry[17] for entry in data3])
    target_psi_deg_rl = np.array([entry[5] for entry in data1])
    ax1[2, 0].plot(time, actual_phi_deg_rl, label="RL")
    ax1[2, 0].plot(time, actual_phi_deg_cascaded_pid, label="Cascaded PID")
    ax1[2, 0].plot(
        time, actual_phi_deg_gain_scheduling_pid, label="Gain-Scheduling PID"
    )
    ax1[2, 0].set_xlabel("Time (s)")
    ax1[2, 0].set_title("Roll Angle (deg)")
    ax1[2, 0].legend()
    ax1[2, 1].plot(time, actual_theta_deg_rl, label="RL")
    ax1[2, 1].plot(time, actual_theta_deg_cascaded_pid, label="Cascaded PID")
    ax1[2, 1].plot(
        time, actual_theta_deg_gain_scheduling_pid, label="Gain-Scheduling PID"
    )
    ax1[2, 1].set_xlabel("Time (s)")
    ax1[2, 1].set_title("Pitch Angle (deg)")
    ax1[2, 0].legend()
    ax1[2, 2].plot(time, actual_psi_deg_rl, label="RL")
    ax1[2, 2].plot(time, actual_psi_deg_cascaded_pid, label="Cascaded PID")
    ax1[2, 2].plot(
        time, actual_psi_deg_gain_scheduling_pid, label="Gain-Scheduling PID"
    )
    ax1[2, 2].plot(time, target_psi_deg_rl, label="Reference")
    ax1[2, 2].set_xlabel("Time (s)")
    ax1[2, 2].set_title("Yaw Angle (deg)")
    ax1[2, 2].legend()

    fig2, ax2 = plt.subplots(3, 2, layout="constrained")
    # Plot the average error in position and orientation
    avg_error_x = np.mean(np.abs(np.array([entry[24] for entry in data1])))
    avg_error_y = np.mean(np.abs(np.array([entry[25] for entry in data1])))
    avg_error_z = np.mean(np.abs(np.array([entry[26] for entry in data1])))

    max_error_x = np.max(np.abs(np.array([entry[24] for entry in data1])))
    max_error_y = np.max(np.abs(np.array([entry[25] for entry in data1])))
    max_error_z = np.max(np.abs(np.array([entry[26] for entry in data1])))

    avg_error_psi_rad = np.mean(np.abs(np.array([entry[27] for entry in data1])))
    avg_error_psi_deg = np.mean(np.abs(np.array([entry[28] for entry in data1])))
    max_error_psi_rad = np.max(np.abs(np.array([entry[27] for entry in data1])))
    max_error_psi_deg = np.max(np.abs(np.array([entry[28] for entry in data1])))

    avg_reward = np.mean(np.array([entry[33] for entry in data1]))

    labels = [
        "X-axis",
        "Y-axis",
        "Z-axis",
        "X-axis",
        "Y-axis",
        "Z-axis",
        "Yaw",
        "Yaw",
        "Yaw",
        "Yaw",
        "Average Reward",
    ]

    values = [
        avg_error_x,
        avg_error_y,
        avg_error_z,
        max_error_x,
        max_error_y,
        max_error_z,
        avg_error_psi_rad,
        max_error_psi_rad,
        avg_error_psi_deg,
        max_error_psi_deg,
    ]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][0:3]

    ax2[0, 0].bar(labels[0:3], values[0:3], color=colors)
    ax2[0, 0].set_title("Avg. Error in Position (m)")
    ax2[0, 1].bar(labels[3:6], values[3:6], color=colors)
    ax2[0, 1].set_title("Max. Error in Position (m)")
    ax2[1, 0].bar(labels[6], values[6], color=colors)
    ax2[1, 0].set_title("Avg. Error in Orientation (rad)")
    ax2[1, 1].bar(labels[7], values[7], color=colors)
    ax2[1, 1].set_title("Max. Error in Orientation (rad)")
    ax2[2, 0].bar(labels[8], values[8], color=colors)
    ax2[2, 0].set_title("Avg. Error in Orientation (deg)")
    ax2[2, 1].bar(labels[9], values[9], color=colors)
    ax2[2, 1].set_title("Max. Error in Orientation (deg)")

    for ax in fig2.axes:
        ax.grid(False)
        for bars in ax.containers:
            ax.bar_label(bars, label_type="center", fmt="%.2f", fontweight="bold")

    fig2.suptitle(
        "Average Reward: {:.2f}".format(avg_reward),
        fontsize=12,
        fontweight="bold",
    )

    fig3, ax3 = plt.subplots(2, 2, layout="constrained")
    # Plot the control signal against time
    motor1_rl = np.array([entry[-5] for entry in data1[:-1]])
    motor1_cascaded_pid = np.array([entry[-5] for entry in data2[:-1]])
    motor1_gain_scheduling_pid = np.array([entry[-5] for entry in data3[:-1]])
    motor2_rl = np.array([entry[-4] for entry in data1[:-1]])
    motor2_cascaded_pid = np.array([entry[-4] for entry in data2[:-1]])
    motor2_gain_scheduling_pid = np.array([entry[-4] for entry in data3[:-1]])
    motor3_rl = np.array([entry[-3] for entry in data1[:-1]])
    motor3_cascaded_pid = np.array([entry[-3] for entry in data2[:-1]])
    motor3_gain_scheduling_pid = np.array([entry[-3] for entry in data3[:-1]])
    motor4_rl = np.array([entry[-2] for entry in data1[:-1]])
    motor4_cascaded_pid = np.array([entry[-2] for entry in data2[:-1]])
    motor4_gain_scheduling_pid = np.array([entry[-2] for entry in data3[:-1]])
    ax3[0, 0].plot(time[:-1], motor1_rl, label="RL", alpha=1)
    ax3[0, 0].plot(time[:-1], motor1_cascaded_pid, label="Cascaded PID", alpha=0.5)
    ax3[0, 0].plot(
        time[:-1], motor1_gain_scheduling_pid, label="Gain-Scheduling PID", alpha=0.7
    )
    ax3[0, 0].set_xlabel("Time (s)")
    ax3[0, 0].set_title("Motor 1 (PWM)")
    ax3[0, 0].legend()
    ax3[0, 1].plot(time[:-1], motor2_rl, label="RL", alpha=1)
    ax3[0, 1].plot(time[:-1], motor2_cascaded_pid, label="Cascaded PID", alpha=0.5)
    ax3[0, 1].plot(
        time[:-1], motor2_gain_scheduling_pid, label="Gain-Scheduling PID", alpha=0.7
    )
    ax3[0, 1].set_xlabel("Time (s)")
    ax3[0, 1].set_title("Motor 2 (PWM)")
    ax3[0, 1].legend()
    ax3[1, 0].plot(time[:-1], motor3_rl, label="RL", alpha=1)
    ax3[1, 0].plot(time[:-1], motor3_cascaded_pid, label="Cascaded PID", alpha=0.5)
    ax3[1, 0].plot(
        time[:-1], motor3_gain_scheduling_pid, label="Gain-Scheduling PID", alpha=0.7
    )
    ax3[1, 0].set_xlabel("Time (s)")
    ax3[1, 0].set_title("Motor 3 (PWM)")
    ax3[1, 0].legend()
    ax3[1, 1].plot(time[:-1], motor4_rl, label="RL", alpha=1)
    ax3[1, 1].plot(time[:-1], motor4_cascaded_pid, label="Cascaded PID", alpha=0.5)
    ax3[1, 1].plot(
        time[:-1], motor4_gain_scheduling_pid, label="Gain-Scheduling PID", alpha=0.7
    )
    ax3[1, 1].set_xlabel("Time (s)")
    ax3[1, 1].set_title("Motor 4 (PWM)")
    ax3[1, 1].legend()

    for ax in fig3.axes:
        ax.grid(False)
        for bars in ax.containers:
            ax.bar_label(bars, label_type="center", fmt="%.2f", fontweight="bold")

    fig3.suptitle(
        "Input Signals",
        fontsize=12,
        fontweight="bold",
    )

    plt.show()

    fig1.set_size_inches(18.5, 10.5)
    fig1.savefig(
        os.path.join(
            "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/examples/evaluation",
            "{}_results.png".format(scenario_name),
        ),
        dpi=800,
    )

    fig2.set_size_inches(18.5, 10.5)
    fig2.savefig(
        os.path.join(
            "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/examples/evaluation",
            "{}_overall_results.png".format(scenario_name),
        ),
        dpi=800,
    )

    fig3.set_size_inches(18.5, 10.5)
    fig3.savefig(
        os.path.join(
            "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/examples/evaluation",
            "{}_motor_inputs.png".format(scenario_name),
        ),
        dpi=800,
    )


def log_episode_slow_traj(scenario_name, data1, data2, data3):
    """
    Write episode log into specified file

        Arguments:
            path_to_log(str): path to directory where to write info
            name(str): name for accumulated log info

    """
    fig1, ax1 = plt.subplots(2, 2, layout="constrained")
    fig2, ax2 = plt.subplots(2, 2, layout="constrained")
    fig3, ax3 = plt.subplots(2, 2, layout="constrained")
    # Plot the control signal against time
    time_rl = np.array([entry[0] for entry in data1]) * 1 / 80
    time_cascaded_pid = np.array([entry[0] for entry in data2]) * 1 / 80
    time_gain_scheduling_pid = np.array([entry[0] for entry in data3]) * 1 / 80
    motor1_rl = np.array([entry[-5] for entry in data1[:-1]])
    motor1_cascaded_pid = np.array([entry[-5] for entry in data2[:-1]])
    motor1_gain_scheduling_pid = np.array([entry[-5] for entry in data3[:-1]])
    motor2_rl = np.array([entry[-4] for entry in data1[:-1]])
    motor2_cascaded_pid = np.array([entry[-4] for entry in data2[:-1]])
    motor2_gain_scheduling_pid = np.array([entry[-4] for entry in data3[:-1]])
    motor3_rl = np.array([entry[-3] for entry in data1[:-1]])
    motor3_cascaded_pid = np.array([entry[-3] for entry in data2[:-1]])
    motor3_gain_scheduling_pid = np.array([entry[-3] for entry in data3[:-1]])
    motor4_rl = np.array([entry[-2] for entry in data1[:-1]])
    motor4_cascaded_pid = np.array([entry[-2] for entry in data2[:-1]])
    motor4_gain_scheduling_pid = np.array([entry[-2] for entry in data3[:-1]])
    ax1[0, 0].plot(time_rl[:-1], motor1_rl, label="RL", alpha=1)
    ax2[0, 0].plot(
        time_cascaded_pid[:-1], motor1_cascaded_pid, label="Cascaded PID", alpha=1
    )
    ax3[0, 0].plot(
        time_gain_scheduling_pid[:-1],
        motor1_gain_scheduling_pid,
        label="Gain-Scheduling PID",
        alpha=1,
    )
    ax1[0, 0].set_xlabel("Time (s)")
    ax2[0, 0].set_xlabel("Time (s)")
    ax3[0, 0].set_xlabel("Time (s)")
    ax1[0, 0].set_title("Motor 1 (PWM)")
    ax2[0, 0].set_title("Motor 1 (PWM)")
    ax3[0, 0].set_title("Motor 1 (PWM)")

    ax1[0, 1].plot(time_rl[:-1], motor2_rl, label="RL", alpha=1)
    ax2[0, 1].plot(
        time_cascaded_pid[:-1], motor2_cascaded_pid, label="Cascaded PID", alpha=1
    )
    ax3[0, 1].plot(
        time_gain_scheduling_pid[:-1],
        motor2_gain_scheduling_pid,
        label="Gain-Scheduling PID",
        alpha=1,
    )
    ax1[0, 1].set_xlabel("Time (s)")
    ax2[0, 1].set_xlabel("Time (s)")
    ax3[0, 1].set_xlabel("Time (s)")
    ax1[0, 1].set_title("Motor 2 (PWM)")
    ax2[0, 1].set_title("Motor 2 (PWM)")
    ax3[0, 1].set_title("Motor 2 (PWM)")

    ax1[1, 0].plot(time_rl[:-1], motor3_rl, label="RL", alpha=1)
    ax2[1, 0].plot(
        time_cascaded_pid[:-1], motor3_cascaded_pid, label="Cascaded PID", alpha=1
    )
    ax3[1, 0].plot(
        time_gain_scheduling_pid[:-1],
        motor3_gain_scheduling_pid,
        label="Gain-Scheduling PID",
        alpha=1,
    )
    ax1[1, 0].set_xlabel("Time (s)")
    ax2[1, 0].set_xlabel("Time (s)")
    ax3[1, 0].set_xlabel("Time (s)")
    ax1[1, 0].set_title("Motor 3 (PWM)")
    ax2[1, 0].set_title("Motor 3 (PWM)")
    ax3[1, 0].set_title("Motor 3 (PWM)")

    ax1[1, 1].plot(time_rl[:-1], motor4_rl, label="RL", alpha=1)
    ax2[1, 1].plot(
        time_cascaded_pid[:-1], motor4_cascaded_pid, label="Cascaded PID", alpha=1
    )
    ax3[1, 1].plot(
        time_gain_scheduling_pid[:-1],
        motor4_gain_scheduling_pid,
        label="Gain-Scheduling PID",
        alpha=1,
    )
    ax1[1, 1].set_xlabel("Time (s)")
    ax2[1, 1].set_xlabel("Time (s)")
    ax3[1, 1].set_xlabel("Time (s)")
    ax1[1, 1].set_title("Motor 4 (PWM)")
    ax2[1, 1].set_title("Motor 4 (PWM)")
    ax3[1, 1].set_title("Motor 4 (PWM)")

    for ax in fig3.axes:
        ax.grid(False)
        for bars in ax.containers:
            ax.bar_label(bars, label_type="center", fmt="%.2f", fontweight="bold")

    fig1.suptitle(
        "RL Input Signals",
        fontsize=12,
        fontweight="bold",
    )

    fig2.suptitle(
        "Cascaded PID Input Signals",
        fontsize=12,
        fontweight="bold",
    )

    fig3.suptitle(
        "Gain-Scheduling PID Input Signals",
        fontsize=12,
        fontweight="bold",
    )

    plt.show()

    fig1.set_size_inches(18.5, 10.5)
    fig1.savefig(
        os.path.join(
            "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/examples/evaluation",
            "{}_rl_motor_inputs.png".format(scenario_name),
        ),
        dpi=800,
    )

    fig2.set_size_inches(18.5, 10.5)
    fig2.savefig(
        os.path.join(
            "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/examples/evaluation",
            "{}_cascaded_pid_motor_inputs.png".format(scenario_name),
        ),
        dpi=800,
    )

    fig3.set_size_inches(18.5, 10.5)
    fig3.savefig(
        os.path.join(
            "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/examples/evaluation",
            "{}_gain_scheduling_pid_motor_inputs.png".format(scenario_name),
        ),
        dpi=800,
    )


def log_episode_fast_traj(data1, data2, data3):
    """
    Write episode log into specified file

        Arguments:
            path_to_log(str): path to directory where to write info
            name(str): name for accumulated log info

    """
    fig1, ax1 = plt.subplots(2, 2, layout="constrained")
    fig2, ax2 = plt.subplots(2, 2, layout="constrained")
    fig3, ax3 = plt.subplots(2, 2, layout="constrained")
    # Plot the control signal against time
    time_rl = np.array([entry[0] for entry in data1]) * 1 / 80
    time_cascaded_pid = np.array([entry[0] for entry in data2]) * 1 / 80
    time_gain_scheduling_pid = np.array([entry[0] for entry in data3]) * 1 / 80
    motor1_rl = np.array([entry[-5] for entry in data1[:-1]])
    motor1_cascaded_pid = np.array([entry[-5] for entry in data2[:-1]])
    motor1_gain_scheduling_pid = np.array([entry[-5] for entry in data3[:-1]])
    motor2_rl = np.array([entry[-4] for entry in data1[:-1]])
    motor2_cascaded_pid = np.array([entry[-4] for entry in data2[:-1]])
    motor2_gain_scheduling_pid = np.array([entry[-4] for entry in data3[:-1]])
    motor3_rl = np.array([entry[-3] for entry in data1[:-1]])
    motor3_cascaded_pid = np.array([entry[-3] for entry in data2[:-1]])
    motor3_gain_scheduling_pid = np.array([entry[-3] for entry in data3[:-1]])
    motor4_rl = np.array([entry[-2] for entry in data1[:-1]])
    motor4_cascaded_pid = np.array([entry[-2] for entry in data2[:-1]])
    motor4_gain_scheduling_pid = np.array([entry[-2] for entry in data3[:-1]])
    ax1[0, 0].plot(time_rl[:-1], motor1_rl, label="RL", alpha=1)
    ax2[0, 0].plot(
        time_cascaded_pid[:-1], motor1_cascaded_pid, label="Cascaded PID", alpha=1
    )
    ax3[0, 0].plot(
        time_gain_scheduling_pid[:-1],
        motor1_gain_scheduling_pid,
        label="Gain-Scheduling PID",
        alpha=1,
    )
    ax1[0, 0].set_xlabel("Time (s)")
    ax2[0, 0].set_xlabel("Time (s)")
    ax3[0, 0].set_xlabel("Time (s)")
    ax1[0, 0].set_title("Motor 1 (PWM)")
    ax2[0, 0].set_title("Motor 1 (PWM)")
    ax3[0, 0].set_title("Motor 1 (PWM)")

    ax1[0, 1].plot(time_rl[:-1], motor2_rl, label="RL", alpha=1)
    ax2[0, 1].plot(
        time_cascaded_pid[:-1], motor2_cascaded_pid, label="Cascaded PID", alpha=1
    )
    ax3[0, 1].plot(
        time_gain_scheduling_pid[:-1],
        motor2_gain_scheduling_pid,
        label="Gain-Scheduling PID",
        alpha=1,
    )
    ax1[0, 1].set_xlabel("Time (s)")
    ax2[0, 1].set_xlabel("Time (s)")
    ax3[0, 1].set_xlabel("Time (s)")
    ax1[0, 1].set_title("Motor 2 (PWM)")
    ax2[0, 1].set_title("Motor 2 (PWM)")
    ax3[0, 1].set_title("Motor 2 (PWM)")

    ax1[1, 0].plot(time_rl[:-1], motor3_rl, label="RL", alpha=1)
    ax2[1, 0].plot(
        time_cascaded_pid[:-1], motor3_cascaded_pid, label="Cascaded PID", alpha=1
    )
    ax3[1, 0].plot(
        time_gain_scheduling_pid[:-1],
        motor3_gain_scheduling_pid,
        label="Gain-Scheduling PID",
        alpha=1,
    )
    ax1[1, 0].set_xlabel("Time (s)")
    ax2[1, 0].set_xlabel("Time (s)")
    ax3[1, 0].set_xlabel("Time (s)")
    ax1[1, 0].set_title("Motor 3 (PWM)")
    ax2[1, 0].set_title("Motor 3 (PWM)")
    ax3[1, 0].set_title("Motor 3 (PWM)")

    ax1[1, 1].plot(time_rl[:-1], motor4_rl, label="RL", alpha=1)
    ax2[1, 1].plot(
        time_cascaded_pid[:-1], motor4_cascaded_pid, label="Cascaded PID", alpha=1
    )
    ax3[1, 1].plot(
        time_gain_scheduling_pid[:-1],
        motor4_gain_scheduling_pid,
        label="Gain-Scheduling PID",
        alpha=1,
    )
    ax1[1, 1].set_xlabel("Time (s)")
    ax2[1, 1].set_xlabel("Time (s)")
    ax3[1, 1].set_xlabel("Time (s)")
    ax1[1, 1].set_title("Motor 4 (PWM)")
    ax2[1, 1].set_title("Motor 4 (PWM)")
    ax3[1, 1].set_title("Motor 4 (PWM)")

    for ax in fig3.axes:
        ax.grid(False)
        for bars in ax.containers:
            ax.bar_label(bars, label_type="center", fmt="%.2f", fontweight="bold")

    # fig1.suptitle(
    #     "RL Input Signals",
    #     fontsize=12,
    #     fontweight="bold",
    # )

    # fig2.suptitle(
    #     "Cascaded PID Input Signals",
    #     fontsize=12,
    #     fontweight="bold",
    # )

    # fig3.suptitle(
    #     "Gain-Scheduling PID Input Signals",
    #     fontsize=12,
    #     fontweight="bold",
    # )

    plt.show()

    fig1.set_size_inches(18.5, 10.5)
    fig1.savefig(
        os.path.join(
            "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/examples/evaluation",
            "fast_trajectory_scenario1_rl_motor_inputs.png",
        ),
        dpi=800,
    )

    fig2.set_size_inches(18.5, 10.5)
    fig2.savefig(
        os.path.join(
            "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/examples/evaluation",
            "fast_trajectory_scenario2_rl_motor_inputs.png",
        ),
        dpi=800,
    )

    fig3.set_size_inches(18.5, 10.5)
    fig3.savefig(
        os.path.join(
            "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/examples/evaluation",
            "fast_trajectory_scenario3_rl_motor_inputs.png",
        ),
        dpi=800,
    )


# Example usage
hovering_scenario1_rl_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/hovering/trained_models/2024_06_14_19_18_13/best_model_23_801_0_25835_723_results/evaluation_results_0.csv"
hovering_scenario1_cascaded_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/hovering/pid_results/evaluation_results_13.csv"
hovering_scenario1_gain_scheduling_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/hovering/pid_results/evaluation_results_14.csv"

hovering_scenario2_rl_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/hovering/trained_models/2024_06_14_19_18_13/best_model_23_801_0_25835_723_results/evaluation_results_1.csv"
hovering_scenario2_cascaded_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/hovering/pid_results/evaluation_results_18.csv"
hovering_scenario2_gain_scheduling_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/hovering/pid_results/evaluation_results_19.csv"

slow_traj_scenario1_rl_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/trajectory_following_slow/trained_models/2024_06_22_04_02_48/best_model_3_2401_0_216293_103542_results/evaluation_results_0.csv"
slow_traj_scenario1_cascaded_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/trajectory_following_slow/pid_results/evaluation_results_8.csv"
slow_traj_scenario1_gain_scheduling_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/trajectory_following_slow/pid_results/evaluation_results_7.csv"

slow_traj_scenario2_rl_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/trajectory_following_slow/trained_models/2024_06_22_04_02_48/best_model_3_2401_0_216293_103542_results/evaluation_results_1.csv"
slow_traj_scenario2_cascaded_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/trajectory_following_slow/pid_results/evaluation_results_9.csv"
slow_traj_scenario2_gain_scheduling_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/trajectory_following_slow/pid_results/evaluation_results_10.csv"

slow_traj_scenario3_rl_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/trajectory_following_slow/trained_models/2024_06_22_04_02_48/best_model_3_2401_0_216293_103542_results/evaluation_results_2.csv"
slow_traj_scenario3_cascaded_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/trajectory_following_slow/pid_results/evaluation_results_11.csv"
slow_traj_scenario3_gain_scheduling_pid_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/gym_envs/quadx_mod_envs/trajectory_following_slow/pid_results/evaluation_results_12.csv"

fast_traj_scenario1_rl_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/trajectory_following_fast/trained_models/2024_06_22_19_02_49/best_model_21_2401_0_38644_2987_results/evaluation_results_6.csv"
fast_traj_scenario2_rl_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/trajectory_following_fast/trained_models/2024_06_22_19_02_49/best_model_21_2401_0_38644_2987_results/evaluation_results_7.csv"
fast_traj_scenario3_rl_file_path = "/home/mchawa/WS/PyFlyt_Fork/PyFlyt/PyFlyt/rl_training/trajectory_following_fast/trained_models/2024_06_22_19_02_49/best_model_21_2401_0_38644_2987_results/evaluation_results_8.csv"

log_episode_fast_traj(
    load_csv(fast_traj_scenario1_rl_file_path),
    load_csv(fast_traj_scenario2_rl_file_path),
    load_csv(fast_traj_scenario3_rl_file_path),
)
