import numpy as np


class missing_variable_error(Exception):
    """raised when necessary variables are not given"""
    def __init__(self, variable_name: str):
        super().__init__(f"{variable_name} has not been specified. Please provide a value.")

class map_maker():
    def __init__(self, map_size: int = None, pixel_size: float = None, pixel_number: int = None, l_degrees: int = None, const: float = 1, index: float = -1):
        ## external variables
        self.map_size = map_size
        self.pixel_size = pixel_size
        self.pixel_number = pixel_number
        self.l_degrees = l_degrees
        self.const = const
        self.index = index
        ## internal variables
        self.spectrum = None                       # unless loaded in
        self.R = None
        self.R_check = False
        self.fs_scale_factor = None
        self.R_fs = None
        self.R_fs_check = False
        self.random_noise_2d = None
        self.random_noise_2d_fs = None
        ## output variables
        self.grf_fs = None
        self.grf = None

    def set_map_variables(self, map_size = None, pixel_size = None, pixel_number = None) -> bool:
        if map_size is not None:
            self.map_size = map_size
        if pixel_size is not None:
            self.pixel_size = pixel_size
        if pixel_number is not None:
            self.pixel_number = pixel_number
        else:
            self.pixel_number = self.map_size*(60/self.pixel_size)
        if not all((self.map_size, self.pixel_size, self.pixel_number)):
            print("!!!!!!!!!!!There are still missing map_variables.")
            return False
        else:
            print("Full set of map variables defined.")
            return True

    def make_map_coordinates(self, map_size = None, pixel_size = None, pixel_number = None):
        
        if any((map_size, pixel_size, pixel_number)) or self.pixel_number is None:
            self.set_map_variables(map_size, pixel_size, pixel_number)
        x = np.linspace(-0.5*self.map_size, 0.5*self.map_size, int(self.pixel_number))
        y = np.linspace(-0.5*self.map_size, 0.5*self.map_size, int(self.pixel_number))
        X, Y = np.meshgrid(x, y)
        self.R = np.sqrt(X**2 + Y**2)
        self.R_check = True
        print("map made.")

        return None
    
    def _compute_fourierSpace_scaler(self) -> float:
        if self.pixel_size is None:
            raise missing_variable_error("pixel_size")
        fs_scale_factor = np.pi/(self.pixel_size/60 * np.pi/180)
        self.fs_scale_factor = fs_scale_factor
        return fs_scale_factor
    
    def _R_fourierSpace_mapping(self) -> list[float]:
        scale = self._compute_fourierSpace_scaler()
        R_fs = self.R*scale
        self.R_fs = R_fs
        self.R_fs_check = True
        return R_fs
    
    def set_spectrum_variables(self, l_degrees: int = None, const:float = None, index: float = None) -> bool:
        if l_degrees is not None:
            print(l_degrees)
            self.l_degrees = l_degrees
            print(self.l_degrees)
        if const is not None:
            self.const = const
        if index is not None:
            self.index = index
        if not any((self.map_size, self.pixel_size, self.pixel_number)):
            print("!!!!!!!!!!!There are still missing variables for the spectrum (power-law).")
            return False
        else:
            print("Full set of variables for the spectrum (power-law) defined.")
            return True

    def make_spectrum(self, const: float = None, l_degrees: int = None, index: float = None) -> None:
            
        self.set_spectrum_variables(const, l_degrees, index)
        l = np.arange(self.l_degrees, dtype=float)

        spectrum=self.const*(l**self.index)
        spectrum[0] = 0
        self.spectrum = spectrum

        return None
    
    def load_spectrum(self, path: str) -> None:
        """ 
            Spectrum different than a power law? Here you go...
            TODO: All of it ...
        """
        return None
    
    def make_spectrum_map(self, spectrum_path: str = None, const: float = None, l_degrees: int = None, index: float = None) -> None:
        """ 
            Creates a map of the spectrum (Inside Fourier Space) as well as a cut version leaving out areas of null values.
            TODO: add option to intervene in scale

        """

        if spectrum_path is not None:
            self.load_spectrum(self, spectrum_path)
        elif any((const, l_degrees, index)) or self.spectrum is None:
            self.make_spectrum(const, l_degrees, index)

        if self.R_fs is None:
            self._R_fourierSpace_mapping()

        spectrum_map_complete = np.zeros(int(self.R_fs.max())+1)
        spectrum_map_complete[0:self.spectrum.size] = self.spectrum

        if self.R_fs is None:
            raise missing_variable_error("R_fs")

        spectrum_map_confined = spectrum_map_complete[(self.R_fs).astype(int)]
        self.spectrum_map_complete = spectrum_map_complete
        self.spectrum_map_confined = spectrum_map_confined
        return None

    def _make_random_noise(self, N: int = None) -> None:
        if N is not None:
            self.pixel_number = N
        self.random_noise_2d = np.random.normal(0, 1, (int(self.pixel_number), int(self.pixel_number)))
        self.random_noise_2d_fs = np.fft.fft2(self.random_noise_2d)
        return None

    def make_gaussian_random_field(self) -> list[float]:
        
        if not self.R_check:
            self.make_map_coordinates()
        if not self.R_fs_check:  
            self._R_fourierSpace_mapping()
        self.make_spectrum_map()
        self._make_random_noise()

        self.grf_fs = self.spectrum_map_confined*self.random_noise_2d_fs              ## gaussian-random-field in Fourier-Space

        self.grf = np.fft.ifft2(np.fft.fftshift(self.grf_fs))           ## gaussian-random-field after inverse fft2

        self.grf_real = np.real(np.fft.ifft2(np.fft.fftshift(self.grf_fs))) 
        
        return self.grf_fs, self.grf, self.grf_real