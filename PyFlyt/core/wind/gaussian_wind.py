import numpy as np

from PyFlyt.core.abstractions import WindFieldClass


# define the wind field
class GaussianWindField(WindFieldClass):
    def __init__(
        self,
        base_wind_velocities=None,
        max_gust_strength=None,
        orn_conv="ENU_FLU",
        np_random: None | np.random.RandomState = None,
    ):
        super().__init__(np_random)
        if base_wind_velocities is None:
            self.base_wind_velocities = np.random.uniform(
                low=[-7.0, -7.0, -2.0], high=[7.0, 7.0, 2.0], size=(3,)
            )
        else:
            self.base_wind_velocities = base_wind_velocities
        if max_gust_strength is None:
            self.max_gust_strength = 7.0
        else:
            self.max_gust_strength = max_gust_strength
        self.time = 0
        self.orn_conv = orn_conv
        self.wind_x = 0
        self.wind_y = 0
        self.wind_z = 0

    def __call__(self, time: float, position: np.ndarray):
        if self.time == 0 or self.time != time:
            self.time = time
            self.wind = np.zeros_like(position)
            self.wind_x = self.base_wind_velocities[0] + np.clip(
                np.random.normal(), -self.max_gust_strength, self.max_gust_strength
            ).round(3)
            self.wind_y = self.base_wind_velocities[1] + np.clip(
                np.random.normal(), -self.max_gust_strength, self.max_gust_strength
            ).round(3)
            self.wind_z = self.base_wind_velocities[2] + np.clip(
                np.random.normal(), -self.max_gust_strength, self.max_gust_strength
            ).round(3)

            if self.orn_conv == "ENU_FLU":
                self.wind[:, 0] = self.wind_x
                self.wind[:, 1] = self.wind_y
                self.wind[:, 2] = self.wind_z
            elif self.orn_conv == "NED_FRD":
                self.wind[:, 0] = self.wind_y
                self.wind[:, 1] = self.wind_x
                self.wind[:, 2] = -self.wind_z
            else:
                raise ValueError("Unknown orientation convention")

        return self.wind
