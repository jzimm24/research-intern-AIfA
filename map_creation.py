import numpy as np
import matplotlib.pyplot as plt
import scipy

import colossus

### TODO: Might be improved if imports take to long
import colossus.cosmology
import colossus.cosmology.cosmology as cosmology
import colossus.halo
import colossus.halo.mass_so as halo_mass
import colossus.halo.concentration as halo_concentration

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# -----------------------------------------------------------------------------------------------------------------------------------------------------

class missing_variable_error(Exception):
    """raised when necessary variables are not given"""
    def __init__(self, variable_name: str):
        super().__init__(f"{variable_name} has not been specified. Please provide a value.")

class missing_internal_variable_error(Exception):
    """raised when necessary variables are calculated"""
    def __init__(self, variable_name: str):
        super().__init__(f"{variable_name} has not been calculated. Please first calculate {variable_name}.")
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------

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
        self.rms = None

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
        X, Y = np.meshgrid(x, y, sparse=True)           ## TODO: check if sparse works correctly
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
    
    def plot_gaussian_random_field(self) -> None:
        im = plt.imshow(self.grf_real, origin='lower', interpolation='bilinear', cmap='RdBu')
        im.set_clim()

        plt.xlabel(r"$\theta_x$[arcmin]")
        plt.ylabel(r"$\theta_y$[arcmin]")

        cbar = plt.colorbar(im)
        cbar.set_label("T [K]")   # label for the scale
        return None
    
    def rms_estimation(self) -> float:
        self.rms = np.sqrt(np.mean(self.grf_real**2))
        return self.rms

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

class lens_profile():
    """ 
        class is mostly reused/cleaned code from Awais Mirza 2019 at Argelander Institut Bonn

        lens is presumed to be NFW
    """
    def __init__(self, cluster_mass = 5e14, cluster_redshift = 0.7, r_max = 10, hubble_constant = 67.74, cosmology = "planck18", omega_m = 0.3089, omega_l = 0.6911, grav_constant = 4.30091 * 10**(-3) / (10**(6)), speed_of_light = scipy.constants.speed_of_light):
        ## cluster parameters
        self.M = cluster_mass
        self.z = cluster_redshift
        ## physics
        self.c = speed_of_light
        self.cosmology = cosmology
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.h_zero = hubble_constant/100 ## automatically scaled
        ## internal parameters
        self.hubble_parameter_factor = None
        self.hubble_parameter = None
        self.comoving_dist = None
        self.r_x = None
        self.c_x = None
        self.rho_crit = None
        self.r_s = None
        self.density_parameter = None
        ## space variables
        self.r_max = r_max ## in Mpc
        self.r_min = 0.1 ## not usually changed
        self.rSpace

        print("cluster mass " + str(self.M) +"\n"
                + "cluster redshift z:" + str(self.z) + "\n"
                + "speed of light:" + str(self.c) + "\n"
                + "scaled hubble constant:" + str(self.h) + "\n"
                + "omega_m:" + str(self.omega_m) + "\n"
                + "omega_l:" + str(self.omega_l) + "\n"
                + "cosmology" + str(self.cosmology))

    def set_rMax(self, r_max: float) -> None:
        self.r_max = r_max
        return None

    def set_rSpace(self, r_min: float = 0.1, r_max: float = None, N: int = 100) -> None:
        if r_max:
            self.set_rMax(r_max)
        self.rSpace = np.linspace(self.r_min, self.r_max, N)
        return None

    def _calc_hubble_parameter_factor(self, omega_m: float = None, omega_l: float = None, z: float = None) -> float:
        if omega_m:
            self.omega_m = omega_m
            print("omega_m changed to: " + str(omega_m))
        if omega_l:
            self.omega_l = omega_l
            print("omega_l changed to: " + str(omega_l))
        if z:
            self.z = z
            print("redshift z changed to: " + str(z))
        self.hubble_parameter_factor = 1/np.sqrt((1+self.z) * self.omega_m + (1+self.z)**4 * self.omega_l)
        #print("Hubble parameter factor is set to: " + str(self.hubble_parameter_factor))
        return self.hubble_parameter_factor

    def _calc_hubble_parameter(self, h_zero: float = None) -> float:
        if h_zero:
            self.h_zero = h_zero
            print("h_zero changed to: " + str(h_zero))
        if any([omega_m: float = None, omega_l: float = None, z: float = None]):
            self._calc_hubble_parameter_factor(omega_m: float = None, omega_l: float = None, z: float = None)

        self.hubble_parameter = np.sqrt(self.h_zero**2 * self.hubble_parameter_factor)
        print("hubble parameter = " + str(self.hubble_parameter))
        return self.hubble_parameter

    def _calc_comoving_dist(self, c: float = None, h_zero: float = None, omega_m: float = None, omega_l: float = None, z: float = None) -> float:
        if h_zero:
            self.h_zero = h_zero
            print("h_zero changed to: " + str(h_zero))
        if c:
            self.c = c
            print("speed of light c changed to: " + str(c))
        if any([omega_m: float = None, omega_l: float = None, z: float = None]):
            self._calc_hubble_parameter_factor(omega_m: float = None, omega_l: float = None, z: float = None)
        
        self.comoving_dist = (self.c * self.h_zero) * integrate.quad(self.hubble_parameter, np.power(1+self.z, -1), 1, args = (self.omega_m, self.omega_l))[0]
        print("comoving_dist = " + str(self.comoving_dist))
    return self.comoving_dist

    def _calc_r_x(self, option_flag: str = "200c") -> float:
    """

    """
    self.r_x = halo_mass.M_to_R(self.M, self.z, option_flag) * self.hubble_parameter /10**3 #!!!!!!!!!!!!!11Using self.hubble_parameter instead of self.h_zero. Is that correct???
    print("r_x = " + str(self.r_x))
    return self.r_x

    def _calc_c_x(self, option_flag: str = "200c", model_flag: str = "diemer15") -> float:
        self.c_x = halo_concentration.concentration(self.M, option_flag, self.z, model = model_flag)
        print("c_x = " + str(self.c_x))
        return self.c_x

    def _calc_rho_crit(self) -> float:
        if not self.hubble_parameter:
            raise missing_internal_variable_error("hubble_parameter")
        self.rho_crit = (3 * self.hubble_parameter**2) / (8 * np.pi * self.G)
    return self.rho_crit

    def _calc_density_parameter(self) -> float:
        self._calc_c_x()
        self.density_parameter = (200/3) * self._calc_rho_crit() * np.power(self.c_x, 3) / (np.log(1+self.c_x) - self.c_x/(1 - self.c_x))
    return self.density_parameter

    def _calc_scale_radius(self) -> float:
        self.r_s = self.r_x/self.c_x
        return self.r_s

    def rho_lens(r, M_200, z):
    """ 
        NFW Density profile in ((M_(solar))/(Mpc**(-3)))
        Input:  - r: float =        
                - M_200: float =    
                - z: float =        
        Return: 
    """
    return self.density_parameter / ((self.rSpace/self.r_s)*(1 + (self.rSpace/self.r_s))**2)

    def M_NFW(r, m, z) -> float:
    return  4 * np.pi * density_parameter(self.M, self.z) * np.power(self.r_s, 3) * (np.log((self.r_s + self.rSpace)/self.r_s) - (r/(self.r_s + self.rSpace)))