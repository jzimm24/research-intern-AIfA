import numpy as np
import matplotlib.pyplot as plt
import scipy

import colossus

### TODO: Might be improved if imports take to long
import colossus.cosmology
import colossus.cosmology.cosmology as cosmo
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

    ## TODO: Decide on wether positions should be in angular position (2d) or in radial distance from the center (radial symmetry)

    def __init__(self, cluster_mass = 5e14, cluster_redshift = 0.7, r_max = 10, hubble_constant = 67.74, cosmology = "planck18", omega_m = 0.3089, omega_l = 0.6911, grav_constant = 4.30091 * 10**(-3) / (10**(6)), speed_of_light = scipy.constants.speed_of_light):
        ## cluster parameters
        self.M = cluster_mass
        self.z = cluster_redshift
        ## physics
        self.c = speed_of_light
        self.G = grav_constant
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

        self.lmb = None
        self.f = None
        self.v_max = None
        ## space variables
        #self.r_max = r_max ## in Mpc
        #self.r_min = 0.1 ## not usually changed
        self.rSpace = None

        self.n = 100

        self.pos_map = None
        ## output
        self.M_nfw = None
        self.rhoLens = None
        self.v_max = None
        self.d_theta = None

        print("cluster mass " + str(self.M) +"\n"
                + "cluster redshift z: " + str(self.z) + "\n"
                + "speed of light: " + str(self.c) + "\n"
                + "scaled hubble constant: " + str(self.h_zero) + "\n"
                + "omega_m: " + str(self.omega_m) + "\n"
                + "omega_l: " + str(self.omega_l) + "\n"
                + "cosmology: " + str(self.cosmology))

        print("############################################")
        
        self._calc_hubble_parameter_factor()

        self.set_angle_map(3, self.n)

        self._calc_comoving_dist()
        self.set_rSpace()

        cosmo.setCosmology(self.cosmology)
        print("Cosmology set as " + self.cosmology)
        return None

    def set_rMax(self, r_max: float) -> None:
        self.r_max = r_max
        return None
    
    def set_rMin(self, r_min: float) -> None:
        self.r_min = r_min
        return None

    def set_angle_map(self, theta_max: float, n: int):
        """       
            so far exclusively squared maps centered at zero
        """
        self.n = n

        dec = np.linspace(-theta_max, theta_max, n)
        asc = np.linspace(theta_max, -theta_max, n)

        pos_map = np.meshgrid(dec, asc)
        self.pos_map = np.array(pos_map)
        print('pos map set!')
        return self.pos_map

    def set_rSpace(self) -> None:  ### naming dist_map
        dx = np.tan(np.max(self.pos_map[0])-np.min(self.pos_map[0])/self.pos_map.shape[1]) * self.comoving_dist
        dy = dx = np.tan(np.max(self.pos_map[1])-np.min(self.pos_map[1])/self.pos_map.shape[2]) * self.comoving_dist

        n = self.n
        x = np.linspace(-n/2, n/2, n) * dx
        y = np.linspace(-n/2, n/2, n) * dy

        X, Y = np.meshgrid(x, y)

        self.rSpace = np.sqrt(X**2 + Y**2)
        return None

    # def set_rSpace(self, r_min: float = 0.1, r_max: float = None, N: int = 100) -> None:
    #     if r_max:
    #         self.set_rMax(r_max)
    #     if r_min:
    #         self.set_rMin(r_min)
    #     self.rSpace = np.linspace(self.r_min, self.r_max, N)
    #     print("rSpace set!")
    #     return None

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

    def _calc_hubble_parameter(self, h_zero: float = None, omega_m: float = None, omega_l: float = None, z: float = None) -> float:
        if h_zero:
            self.h_zero = h_zero/100
            print("h_zero changed to: " + str(h_zero))
        if any([omega_m, omega_l, z]):
            self._calc_hubble_parameter_factor(omega_m, omega_l, z)

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
        if any([omega_m, omega_l, z]):
            self._calc_hubble_parameter_factor(omega_m, omega_l, z)
        
        self.comoving_dist = (self.c * self.h_zero) * scipy.integrate.quad(lambda a: 1/np.sqrt(a * self.omega_m + a**4 * self.omega_l), 1/(1+self.z), 1)[0]
        print("comoving_dist = " + str(self.comoving_dist))
        return self.comoving_dist

    def _calc_r_x(self, option_flag: str = "200c", m: float = None, z: float = None, h_zero: float = None, omega_m: float = None, omega_l: float = None) -> float:
        if any([m]):
            self.M = m
        if any([z]):
            self.z = z
        if any([h_zero, omega_m, omega_l, z]) or not self.hubble_parameter:
            self._calc_hubble_parameter(h_zero, omega_m, omega_l, z)
        self.r_x = halo_mass.M_to_R(self.M, self.z, option_flag) * self.hubble_parameter /10**3 #!!!!!!!!!!!!!11Using self.hubble_parameter instead of self.h_zero. Is that correct???
        print("r_x = " + str(self.r_x))
        return self.r_x

    def _calc_c_x(self, option_flag: str = "200c", model_flag: str = "diemer15", m: float = None, z: float = None) -> float:
        if any([m]):
            self.M = m
        if any([z]):
            self.z = z
        self.c_x = halo_concentration.concentration(self.M, option_flag, self.z, model = model_flag)
        print("c_x = " + str(self.c_x))
        return self.c_x  

    def _calc_rho_crit(self, h_zero: float = None, omega_m: float = None, omega_l: float = None, z: float = None) -> float:
        if not self.hubble_parameter:
            raise missing_internal_variable_error("hubble_parameter")
        if any([h_zero, omega_m, omega_l, z]):
            self._calc_hubble_parameter(h_zero, omega_m, omega_l, z)
        self.rho_crit = (3 * self.hubble_parameter**2) / (8 * np.pi * self.G)
        print("rho_crit = " + str(self.rho_crit))
        return self.rho_crit

    def _calc_density_parameter(self, h_zero: float = None, omega_m: float = None, omega_l: float = None, m: float = None, z: float = None, option_flag: str = "200c", model_flag: str = "diemer15") -> float:
        new_option_flag = bool("200c" != option_flag)
        new_model_flag = bool("diemer15" != model_flag)
        if any([new_option_flag, new_model_flag, m, z]) or not self.c_x:
            self._calc_c_x(option_flag, model_flag, m, z)
        if any([h_zero, omega_m, omega_l, z]) or not self.rho_crit:
            self._calc_rho_crit(h_zero, omega_m, omega_l, z)
        if not self.hubble_parameter:
            raise missing_internal_variable_error("hubble_parameter")
            #self._calc_rho_crit()
        self.density_parameter = (200/3) * self.rho_crit * np.power(self.c_x, 3) / (np.log(1+self.c_x) - self.c_x/(1 - self.c_x))
        print("density_parameter: " + str(self.density_parameter))
        return self.density_parameter

    def _calc_scale_radius(self, option_flag: str = "200c", model_flag: str = "diemer15", m: float = None, z: float = None) -> float:
        new_option_flag = bool("200c" != option_flag)
        new_model_flag = bool("diemer15" != model_flag)
        if any([new_option_flag, new_model_flag, m, z]) or not self.c_x:
            self._calc_c_x(option_flag, model_flag, m, z)
        if any([new_option_flag, m, z]):
            self._calc_r_x(option_flag, m, z)
        if not self.r_x:
            self._calc_r_x(option_flag, m, z)
        self.r_s = self.r_x/self.c_x
        print("scale radius r_s: " + str(self.r_s))

        return self.r_s

    def rho_lens(self, h_zero: float = None, omega_m: float = None, omega_l: float = None, m: float = None, z: float = None, option_flag: str = "200c", model_flag: str = "diemer15") -> list[float]:
        """ 
            NFW Density profile in ((M_(solar))/(Mpc**(-3)))
            Input:  - r: float =        
                    - M_200: float =    
                    - z: float =        
            Return: 
        """
        new_option_flag = bool("200c" != option_flag)
        new_model_flag = bool("diemer15" != model_flag)
        if any([h_zero, omega_m, omega_l, m, z, new_option_flag, new_model_flag]):
            self._calc_density_parameter(h_zero, omega_m, omega_l, m, z, option_flag, model_flag)
        if any([new_option_flag, new_model_flag, m, z]):
            self._calc_scale_radius(option_flag, model_flag, m, z)
        self.rhoLens = self.density_parameter / ((self.rSpace/self.r_s)*(1 + (self.rSpace/self.r_s))**2)
        return self.rhoLens

    def M_NFW(self, h_zero: float = None, omega_m: float = None, omega_l: float = None, m: float = None, z: float = None, option_flag: str = "200c", model_flag: str = "diemer15") -> list[float]:
        new_option_flag = bool("200c" != option_flag)
        new_model_flag = bool("diemer15" != model_flag)
        if any([h_zero, omega_m, omega_l, m, z, new_option_flag, new_model_flag]):
            self._calc_density_parameter(h_zero, omega_m, omega_l, m, z, option_flag, model_flag)
        if any([new_option_flag, new_model_flag, m, z]):
            self._calc_scale_radius(option_flag, model_flag, m, z)
        self.M_nfw = 4 * np.pi * self.density_parameter * np.power(self.r_s, 3) * (np.log((self.r_s + self.rSpace)/self.r_s) - (self.rSpace/(self.r_s + self.rSpace)))
        return  self.M_nfw

    def set_angle_map(self, theta_max: float, n: int):
        """       
            so far exclusively squared maps centered at zero
        """
        dec = np.linspace(-theta_max, theta_max, n)
        asc = np.linspace(theta_max, -theta_max, n)

        pos_map = np.meshgrid(dec, asc)
        self.pos_map = np.array(pos_map)
        print('pos map set!')
        return self.pos_map

    def _calc_lambda(self, angles: list[list[float]] = None, c: float = None, h_zero: float = None, omega_m: float = None, omega_l: float = None, z: float = None, option_flag: str = "200c", model_flag: str = "diemer15", m: float = None) -> list[list[float]]:
        new_option_flag = bool("200c" != option_flag)
        new_model_flag = bool("diemer15" != model_flag)
        if angles or not self.c_x:
            self.pos_map = angles
        if any([c, h_zero, omega_m, omega_l, z]):
            self._calc_comoving_dist(c, h_zero, omega_m, omega_l, z)
        if z:
            self.z = z
        if any([new_option_flag, new_model_flag, m, z]):
            self._calc_scale_radius(option_flag, model_flag, m, z)
        #### TODO: decide on position map
        radial_pos = np.sqrt(np.add(self.pos_map[0]**2, self.pos_map[1]**2))
        ####
        self.lmb = self.comoving_dist * radial_pos / ((1+self.z) * self.r_s)
        print('lambda calculated!')
        return self.lmb

    def _calc_f(self, angles: list[list[float]] = None, c: float = None, h_zero: float = None, omega_m: float = None, omega_l: float = None, z: float = None, option_flag: str = "200c", model_flag: str = "diemer15", m: float = None):
        new_option_flag = bool("200c" != option_flag)
        new_model_flag = bool("diemer15" != model_flag)
        if any([angles, c, h_zero, omega_m, omega_l, z, new_option_flag, new_model_flag, m]) or not self.lmb:
            self._calc_lambda(angles, c, h_zero, omega_m, omega_l, z, option_flag, model_flag, m)
        if angles:
            self.pos_map = angles
        
        #### TODO: decide on position map
        radial_pos = np.sqrt(np.add(self.pos_map[0]**2, self.pos_map[1]**2))
        ####

        f = np.zeros(np.shape(radial_pos))
        for i in range(np.shape(f)[0]):
            for j in range(np.shape(f)[1]):
                if self.lmb[i][j] < 1:
                    f[i][j] = (3.23 / self.lmb[i][j]) * (np.log(self.lmb[i][j]/2)) + np.log(self.lmb[i][j]/(1 - np.sqrt(1 - self.lmb[i][j]**2))) / np.sqrt(1 - self.lmb[i][j]**2)
                elif self.lmb[i][j] > 1:
                    f[i][j] = (3.23 / self.lmb[i][j]) * (np.log(self.lmb[i][j]/2) + (np.pi/2 - np.arcsin(1/self.lmb[i][j]) ) / np.sqrt(self.lmb[i][j]**2 - 1) )
                else:
                    print("lmb = " + str(self.lmb[i][j]) + "lmb will be calculated as if smaller than 1")
        self.f = f
        print("f calculated!")
        return self.f

    def _calc_v_max(self, option_flag: str = "200c", model_flag: str = "diemer15", m: float = None, z: float = None, h_zero: float = None, omega_m: float = None, omega_l: float = None):
        new_option_flag = bool("200c" != option_flag)
        new_model_flag = bool("diemer15" != model_flag)
        if any([new_option_flag, new_model_flag, m, z]) or self.r_s is None:
            self._calc_scale_radius(option_flag, model_flag, m, z)
        if any([h_zero, omega_m, omega_l, m, z, new_option_flag, new_model_flag]) or self.density_parameter is None:
            self._calc_density_parameter(h_zero, omega_m, omega_l, m, z, option_flag, model_flag)
        self.v_max = 0.46 * np.sqrt(4 * np.pi * self.G * self.density_parameter * self.r_s**2)
        print("v_max calculated!")
        return self.v_max

    def delta_theta(self, r_min: float = None, r_max: float = None, N: int = None, c: float = None, h_zero: float = None, omega_m: float = None, omega_l: float = None, z: float = None, m: float = None, option_flag: str = "200c", model_flag: str = "diemer15", angles: list[list[float]] = None) -> None:
        new_option_flag = bool("200c" != option_flag)
        new_model_flag = bool("diemer15" != model_flag)
        if any([r_min, r_max, N]):
            self.set_rSpace()
        if any([c, h_zero, omega_m, omega_l, z]) or not self.comoving_dist:
            self._calc_comoving_dist(c, h_zero, omega_m, omega_l, z)
        if any([new_option_flag, new_model_flag, m, z, h_zero, omega_m, omega_l]) or not self.v_max:
            self._calc_v_max(option_flag, model_flag, m, z, h_zero, omega_m, omega_l)
        if any([angles, c, h_zero, omega_m, omega_l, z, new_option_flag, new_model_flag, m]) or not self.f:
            self._calc_f(angles, c, h_zero, omega_m, omega_l, z, option_flag, model_flag, m)

        theta = self.rSpace/self.comoving_dist
        print("theta: " + str(theta))                                           ## what map should one take
        #d_theta = 0.54 * (self.v_max)**2 * (self.comoving_dist - self._calc_comoving_dist(z=1100.0)) * (self._calc_f(angles=theta) / self._calc_comoving_dist(z = 1100.0))
        self.d_theta = 0.54 * (self.v_max)**2 * (self.comoving_dist - self._calc_comoving_dist(z=1100.0)) * (self.f / self._calc_comoving_dist(z = 1100.0))
        print("angular displacement calculated!")
        print("Done!")
        return None
    
    def plot_d_theta(self) -> None:
        if not self.d_theta.any():
            raise missing_internal_variable_error("d_theta")
        im = plt.imshow(self.d_theta, origin='lower', interpolation='bilinear', cmap='RdBu')
        im.set_clim()

        plt.xlabel(r"$\Delta\theta_x$[arcmin]")
        plt.ylabel(r"$\Delta\theta_y$[arcmin]")

        cbar = plt.colorbar(im)
        cbar.set_label("T [K]")   # label for the scale
        return None