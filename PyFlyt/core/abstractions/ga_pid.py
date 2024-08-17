import numpy as np

uss = np.array([0.365, 0.0, 0.0, 0.0], dtype=np.float32).reshape(-1, 1)

k_1_neg45_pos45 = np.array(
    [
        [0, 0, -0.05, 0, 0, 0, 0, 0, -0.08, 0, 0, 0],
        [0, 0.02, 0, 0.2, 0, 0, 0, 0.04, 0, 0.01, 0, 0],
        [-0.02, 0, 0, 0, 0.2, 0, -0.04, 0, 0, 0, 0.01, 0],
        [0, 0, 0, 0, 0, 0.07, 0, 0, 0, 0, 0, 0.08],
    ],
    dtype=np.float32,
)

k_1_pos45_pos135 = np.array(
    [
        [0, 0, -0.05, 0, 0, 0, 0, 0, -0.08, 0, 0, 0],
        [-0.02, 0, 0, 0.2, 0, 0, 0, 0.04, 0, 0.01, 0, 0],
        [0, -0.02, 0, 0, 0.2, 0, -0.04, 0, 0, 0, 0.01, 0],
        [0, 0, 0, 0, 0, 0.07, 0, 0, 0, 0, 0, 0.08],
    ],
    dtype=np.float32,
)

k_1_neg45_neg135 = np.array(
    [
        [0, 0, -0.05, 0, 0, 0, 0, 0, -0.08, 0, 0, 0],
        [0.02, 0, 0, 0.2, 0, 0, 0, 0.04, 0, 0.01, 0, 0],
        [0, 0.02, 0, 0, 0.2, 0, -0.04, 0, 0, 0, 0.01, 0],
        [0, 0, 0, 0, 0, 0.07, 0, 0, 0, 0, 0, 0.08],
    ],
    dtype=np.float32,
)

k_1_neg135_pos135 = np.array(
    [
        [0, 0, -0.05, 0, 0, 0, 0, 0, -0.08, 0, 0, 0],
        [0, -0.02, 0, 0.2, 0, 0, 0, 0.04, 0, 0.01, 0, 0],
        [0.02, 0, 0, 0, 0.2, 0, -0.04, 0, 0, 0, 0.01, 0],
        [0, 0, 0, 0, 0, 0.07, 0, 0, 0, 0, 0, 0.08],
    ],
    dtype=np.float32,
)

c = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)


def ga_pid_step(state: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
    # Create a Gain Scheduling PID Controller
    if not hasattr(ga_pid_step, "k_1"):
        ga_pid_step.k_1 = k_1_neg45_pos45  # Initialize the static variable
    state = state.flatten()
    state[3:6] = (state[3:6] + np.pi) % (2 * np.pi) - np.pi
    new_state = np.array([state[9:12], state[3:6], state[6:9], state[0:3]]).reshape(
        -1, 1
    )
    setpoint = setpoint.reshape(-1)
    setpoint[2] = (setpoint[2] + np.pi) % (2 * np.pi) - np.pi
    setpoint = np.array([setpoint[0], setpoint[1], setpoint[3], setpoint[2]]).reshape(
        -1, 1
    )
    xss = np.transpose(c) @ setpoint
    if new_state[5] >= -0.7854 and new_state[5] <= 0.785398:
        ga_pid_step.k_1 = k_1_neg45_pos45
    elif new_state[5] > 0.785398 and new_state[5] <= 2.35619:
        ga_pid_step.k_1 = k_1_pos45_pos135
    elif new_state[5] < -0.7854 and new_state[5] >= -2.35619:
        ga_pid_step.k_1 = k_1_neg45_neg135
    else:
        ga_pid_step.k_1 = k_1_neg135_pos135

    error = new_state - xss

    output = (-ga_pid_step.k_1 @ (error)) + uss
    # output = uss
    output = np.asarray(output).reshape(-1)
    output = np.array([output[1], output[2], output[3], output[0]]).reshape(-1)

    # print(
    #     "X-Pos: {}, X-Vel Correction: {}".format(
    #         (new_state[0]), (-ga_pid_step.k_1[1][0] * error[0])
    #     )
    # )
    # print(output[1])

    return output


if __name__ == "__main__":
    state = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    setpoint = np.array([[0, 0, 0, 0]])
    print(ga_pid_step(state, setpoint))
