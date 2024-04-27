import numpy as np
from stable_baselines3.common.type_aliases import PolicyPredictor


class HoveringPIDExpert(PolicyPredictor):
    def __init__(self, taget_pos, target_psi):
        self.target_pos = taget_pos
        self.target_psi = target_psi

        self.set_point = np.array(
            [
                self.target_pos[0],
                self.target_pos[1],
                self.target_psi,
                self.target_pos[2],
            ],
            ndmin=2,
        )

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
    ):

        return (self.set_point, state)
