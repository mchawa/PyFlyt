import numpy as np

from PyFlyt.core.abstractions import WindFieldClass


# define the wind field
class SimpleWindField(WindFieldClass):
    def __init__(
        self, my_parameter=1.0, np_random: None | np.random.RandomState = None
    ):
        super().__init__(np_random)
        self.strength = my_parameter

    def __call__(self, time: float, position: np.ndarray):
        # simulate a thermal windfield, where the xy velocities are 0,
        wind = np.zeros_like(position)
        # but the z velocity varies to the log of height,
        height = np.clip(position[:, -1] + 1, 0, None)
        wind[:, -1] = np.log(height) * self.strength
        # plus some noise,
        wind += self.np_random.randn(*wind.shape)
        return wind
