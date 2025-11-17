import numpy as np


class map_maker():
    def __init__(self, map_size: int = None, pixel_size: float = None, pixel_number: int = None, l_degrees: int = None, const: float = 1, index: float = -1):
        self.map_size = map_size
        self.pixel_size = pixel_size
        self.pixel_number = pixel_number
        self.l_degrees = l_degrees
        self.const = const
        self.index = index

    def set_map_variables(self, map_size = None, pixel_size = None, pixel_number = None) -> bool:
        if map_size is not None:
            self.map_size = map_size
        if pixel_size is not None:
            self.pixel_size = pixel_size
        if pixel_number is not None:
            self.pixel_number = pixel_number
        else:
            self.pixel_number = self.map_size*(60/self.pixel_size)
        if not any((self.map_size, self.pixel_size, self.pixel_number)):
            print("!!!!!!!!!!!There are still missing map_variables")
            return False
        else:
            print("Full set of map variables defined")
            return True

    def make_spectrum(self, map_size = None, pixel_size = None, pixel_number = None):
        
        if any((map_size, pixel_size, pixel_number)):
            self.set_map_variables(map_size, pixel_size, pixel_number)

        x = np.linspace(-0.5*map_size, 0.5*map_size, int(pixel_number))
        y = np.linspace(-0.5*map_size, 0.5*map_size, int(pixel_number))
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)

        l = np.arange(30000, dtype=float)
        const = 10
        index = -0.7

        spectrum=const*(l**index)
        spectrum[0] = 0

    def make_noise():
        return None

    def make_gaussian_random_field():
        return None