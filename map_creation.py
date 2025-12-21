import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.interpolate import interp1d

import colossus

### TODO: Might be improved if imports take to long
import colossus.cosmology
import colossus.cosmology.cosmology as cosmo
import colossus.halo
import colossus.halo.mass_so as halo_mass
import colossus.halo.concentration as halo_concentration

from astropy import constants as const
from astropy import units as u
from astropy import coordinates as coord
from astropy.cosmology import FlatLambdaCDM

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

class missing_spectrum_type_error(Exception):
    """raised when spectrum type is not given"""
    def __init__(self):
        super().__init__(f"Spectrum type has not been specified. Please provide a type.")
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------

class map_maker():
    def __init__(self, map_size: int = None, pixel_size: float = None, pixel_number: int = None, l_degrees: int = None, const: float = 1, index: float = -1, spectrum_type: str = "power law"):
        ## external variables
        self.map_size = map_size
        self.pixel_size = pixel_size
        self.pixel_number = pixel_number
        self.l_degrees = l_degrees
        self.const = const
        self.index = index
        self.spectrum_type = spectrum_type
        ## internal variables
        self.spectrum = None                       # unless loaded in

        self.X = None
        self.Y = None
        self.X_fs = None
        self.Y_fs = None

        self.X_scaled = None
        self.Y_scaled = None
        self.X_fs_scaled = None
        self.Y_fs_scaled = None

        self.R = None
        self.R_scaled = None   
        self.R_check = False
        self.fs_scale_factor = None
        self.R_fs = None
        self.R_fs_scaled = None
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
            self.pixel_number = self.map_size/(self.pixel_size)
        if not all((self.map_size, self.pixel_size, self.pixel_number)):
            print("!!!!!!!!!!!There are still missing map_variables.")
            return False
        else:
            print("Full set of map variables defined.")
            print("Map Size: " + str(self.map_size) + "\n"
                  + "Pixel Size: " + str(self.pixel_size) + "\n"
                  + "Number of Pixel: " + str(self.pixel_number))
            return True

    # def make_map_coordinates(self, map_size = None, pixel_size = None, pixel_number = None):
        
    #     if any((map_size, pixel_size, pixel_number)) or self.pixel_number is None:
    #         self.set_map_variables(map_size, pixel_size, pixel_number)
    #     x = np.linspace(-0.5*self.map_size, 0.5*self.map_size, int(self.pixel_number))
    #     y = np.linspace(-0.5*self.map_size, 0.5*self.map_size, int(self.pixel_number))
    #     X, Y = np.meshgrid(x, y, sparse=True)           ## TODO: check if sparse works correctly
    #     self.R = np.sqrt(X**2 + Y**2)
    #     self.R_check = True
    #     print("coordinates made.")
    #     self.X = X
    #     self.Y = Y

    #     return None

    def make_map_coordinates(self, pixel_size: float = None, pixel_number: int = None) -> None:
        if pixel_number:
            self.pixel_number = pixel_number
        if pixel_size:
            self.pixel_size = pixel_size
        if self.pixel_number is None or self.pixel_size is None:
            raise ValueError("pixel_number and pixel_size must be set")

        # integer-centered coordinates
        x, y = (np.arange(self.pixel_number) - self.pixel_number // 2) * self.pixel_size, (np.arange(self.pixel_number) - self.pixel_number // 2) * self.pixel_size

        X, Y = np.meshgrid(x, y, indexing="xy")

        self.X = X
        self.Y = Y
        self.R = np.sqrt(X**2 + Y**2)

        self.R_check = True
        print("Real-space coordinates made (FFT-safe).")
        return None
    
    def make_fourier_map_coordinates(self) -> None:
        self._compute_fourierSpace_scaler()
        #pixel_size_fs = (np.pi/180)*(self.pixel_size/60.)
        #x_fs = 2*np.pi(np.fft.fftfreq(self.pixel_number, pixel_size_fs))
        #y_fs = 2*np.pi(np.fft.fftfreq(self.pixel_number, pixel_size_fs))
        pixel_size_fs = 2*np.pi*self.fs_scale_factor
        x_fs = 2*np.pi(np.fft.fftfreq(self.pixel_number, pixel_size_fs))
        y_fs = 2*np.pi(np.fft.fftfreq(self.pixel_number, pixel_size_fs))
        #x_fs_min, x_fs_max = min(x_fs), max(x_fs)
        #y_fs_min, y_fs_max = min(y_fs), max(y_fs)
        X_fs, Y_fs = np.meshgrid(x_fs, y_fs)
        self.X_fs, self.Y_fs = X_fs, Y_fs
        return None
    
    def _normalize_coordinates(self, X: list[float] = None, Y: list[float] = None, Fourier=False) -> None:
        
        if Fourier:
            self.X_fs_scaled = self.X/(np.max(self.X)-np.min(self.X))
            self.Y_fs_scaled = self.Y/(np.max(self.Y)-np.min(self.Y))
            self.R_fs_scaled = np.sqrt(self.X_fs_scaled**2 + self.Y_fs_scaled**2)
        else:
            #self.X_scaled = self.X/(np.max(self.X[0])-np.min(self.X[0]))*2
            self.X_scaled = self.X/(np.max(self.X)-np.min(self.X))*2
            self.Y_scaled = self.Y/(np.max(self.Y)-np.min(self.Y))*2
            self.R_scaled = np.sqrt(self.X_scaled**2 + self.Y_scaled**2)
        print("Normalized Coordinates set.")
        return None

    
    def _compute_fourierSpace_scaler(self) -> float:
        if self.pixel_size is None:
            raise missing_variable_error("pixel_size")
        fs_scale_factor = (self.pixel_size/60 * np.pi/180)
        #fs_scale_factor = 2*np.pi*(self.pixel_size/60) * (np.pi/180)            ## TODO: Check if aarcmin/arcsec conversion is correct here
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

    def make_spectrum(self, const: float = None, l_degrees: int = None, index: float = None, zero_dipole=False, spectrum_type: str = None,) -> None:
            
        self.set_spectrum_variables(const, l_degrees, index)
        l = np.arange(2, self.l_degrees, dtype=float)
        if spectrum_type:
            self.spectrum_type = spectrum_type
        if self.spectrum_type == "power law":
            spectrum=self.const*(l**self.index)
        if self.spectrum_type == "acoustic":
            print("Used Spectrum type acoustic.")
            Dl = 6000 * (l / 200.)**(-1) * (1 + 0.5 * np.sin(l / 200. - 2) * np.exp(-(l - 200)**2 / 50000))
            spectrum = Dl * 2 * np.pi / (l * (l + 1.))
        else:
            raise missing_spectrum_type_error()
        spectrum[0] = 0
        if zero_dipole:
            spectrum[1] = 0
            #spectrum[2] = 0
        self.spectrum = spectrum

        return None
    
    def load_and_make_spectrum(self, spectrum_path, zero_dipole = False) -> None:
        l, Dl = np.loadtxt(spectrum_path, 
                             usecols=(0, 1), unpack=True)
        spectrum = Dl * 2 * np.pi / (l * (l + 1.))
        spectrum[0] = 0
        if zero_dipole:
            spectrum[1] = 0
            #spectrum[2] = 0
        self.spectrum = spectrum
        return None
    
    def load_spectrum(self, path: str) -> None:
        """ 
            Spectrum different than a power law? Here you go...
            TODO: All of it ...
        """
        return None
    
    def make_spectrum_map(self, spectrum_path: str = None, const: float = None, l_degrees: int = None, index: float = None, zero_dipole = False, spectrum_type = "power law") -> None:
        """ 
            Creates a map of the spectrum (Inside Fourier Space) as well as a cut version leaving out areas of null values.
            TODO: add option to intervene in scale

        """
        self._normalize_coordinates()
        if spectrum_path is not None:
            self.load_spectrum(self, spectrum_path)
        elif any((const, l_degrees, index)) or self.spectrum is None:
            self.make_spectrum(const, l_degrees, index, zero_dipole, spectrum_type)

        multipole_field = self.R_scaled * np.pi / self.fs_scale_factor

        #spectrum_map_complete = np.zeros(int(multipole_field.max())+1)
        #spectrum_map_complete[0:self.spectrum.size] = self.spectrum

        # if self.R_fs is None:
        #     raise missing_variable_error("R_fs")

        #spectrum_map_confined = spectrum_map_complete[multipole_field.astype(int)]
        spectrum_interp = interp1d(
        np.arange(len(self.spectrum)),  # l values
        self.spectrum,                  # C_l values
        kind='linear',                  # linear interpolation
        bounds_error=False,             # allow extrapolation
        fill_value=0                    # values outside get zero
        )

        # Evaluate the spectrum at each point in Fourier space
        spectrum_map_confined = spectrum_interp(multipole_field)
        
        #self.spectrum_map_complete = spectrum_map_complete
        self.spectrum_map_confined = spectrum_map_confined
        return None

    def _make_random_noise(self, N: int = None) -> None:
        if N is not None:
            self.pixel_number = N
        self.random_noise_2d = np.random.normal(0, 1, (int(self.pixel_number), int(self.pixel_number)))
        self.random_noise_2d_fs = np.fft.fft2(self.random_noise_2d)
        return None

    def make_gaussian_random_field(self, zero_dipole = False, spectrum_type: str = "power law") -> list[float]:
        
        if not self.R_check:
            self.make_map_coordinates()
        if not self.R_fs_check:  
            self._R_fourierSpace_mapping()
        self.make_spectrum_map(zero_dipole=zero_dipole, spectrum_type=spectrum_type)
        self._make_random_noise()

        self.grf_fs = np.sqrt(self.spectrum_map_confined)*self.random_noise_2d_fs              ## gaussian-random-field in Fourier-Space

        # enforce Hermitian symmetry explicitly
        N = int(self.pixel_number)
        for i in range(N):
            for j in range(N):
                ii = (-i) % N
                jj = (-j) % N
                self.grf_fs[ii, jj] = np.conj(self.grf_fs[i, j])

        #self.grf = np.fft.ifft2(np.fft.fftshift(self.grf_fs)) / self.fs_scale_factor      ## gaussian-random-field after inverse fft2
        self.grf = np.fft.ifft2(np.fft.ifftshift(self.grf_fs)) / self.fs_scale_factor
        #self.grf = np.fft.ifft2(self.grf_fs) / self.fs_scale_factor

        #self.grf_real = np.real(np.fft.ifft2(np.fft.fftshift(self.grf_fs))) / self.fs_scale_factor
        self.grf_real = np.real(np.fft.ifft2(np.fft.ifftshift(self.grf_fs))) / self.fs_scale_factor
        #self.grf_real = np.real(np.fft.ifft2(self.grf_fs)) / self.fs_scale_factor
        
        return self.grf_fs, self.grf, self.grf_real
    
    def plot_gaussian_random_field(self) -> None:
        im = plt.imshow(self.grf_real, extent = [0, self.pixel_number*self.pixel_size, 0, self.pixel_number*self.pixel_size], origin='lower', interpolation='bilinear', cmap='RdBu')
        im.set_clim()

        plt.xlabel(r"$\theta_x$[arcmin]")
        plt.ylabel(r"$\theta_y$[arcmin]")

        cbar = plt.colorbar(im)
        cbar.set_label("T [μK]")   # label for the scale
        return None
    
    def rms_estimation(self) -> float:
        self.rms = np.sqrt(np.mean(self.grf_real**2))
        return self.rms

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

class nfw_lens():
    def __init__(self, cluster_mass: float = 8e14, cluster_redshift: float = 0.5, map_size: float = None, pixel_size: float = None, pixel_number: int = None, rho_type: str = "crit", cosmology = FlatLambdaCDM(H0=67.74, Om0=0.3089), h_0: float = None, omega_m: float = None, grav_constant = 4.30091 * 10**(-3) / (10**(6)), speed_of_light = scipy.constants.speed_of_light, delta = 500, z_source = 1100):
        ## cluster parameters
        self.M = cluster_mass * u.Msun
        self.z = cluster_redshift
        ## physics
        self.z_source = z_source
        self.c = speed_of_light
        self.G = grav_constant

        self.cosmology = cosmology
        self.delta = delta
        self.h = h_0
        self.omega_m = omega_m
        ## external variables
        self.map_size = map_size
        self.pixel_size = pixel_size
        self.pixel_number = pixel_number

        self.rho_type = rho_type
        ## internal variables
        self.map = None             # !!! in degrees; only centered and squared TODO: more flexible maps
        self.X_arcmin = None
        self.Y_arcmin = None
        self.X_deg = None
        self.Y_deg = None

        self.cluster_coords = coord.SkyCoord(ra=0 * u.degree, dec=0 * u.degree)     ## hardcoded central galaxy cluster TODO: add more flexible cluster positions

        self.ang_sep = None
        self.rho_c_z = None
        self.r_v = None
        self.concentration = None
        self.nfwConcentration = None

        self.r_s = None

        self.D_l = None
        self.D_s = None
        self.D_ls = None

        self.sigma_c = None

        self.physical_r = None
        self.physical_r_scaled = None
        self.physical_r_scaledMin = 1e-4
        self.center_sing_extension = None

        self.lX = None
        self.lY = None
        self.l2d = None
        self.l2d_check = False

        ## output
        self.proj = None
        self.sigma = None
        self.Kappa = None

        self.alphaX = None
        self.alphaY = None
        self.alphaX_fft = None
        self.alphaY_fft = None

        if map_size and pixel_size:
            self.pixel_number = map_size / pixel_size
        if map_size and pixel_number:
            self.pixel_size = map_size / pixel_number
        if pixel_size and pixel_number:
            self.map_size = pixel_number*pixel_size

        if any([h_0, omega_m]):  
            self.cosmology = FlatLambdaCDM(H0=h_0, Om0=omega_m)

        cosmo.setCosmology("planck15")

        return None
        

    def make_map(self, map_size: float = None, pixel_size: float = None, pixel_number: int = None):
        if map_size and pixel_size:
            self.map_size = map_size
            self.pixel_size = pixel_size
            self.pixel_number = int(self.map_size / self.pixel_size)
        elif pixel_size and pixel_number:
            self.pixel_size = pixel_size
            self.pixel_number = pixel_number
            self.map_size = self.pixel_number * self.pixel_size
        elif pixel_number and map_size:
            self.pixel_number = pixel_number
            self.map_size = map_size
            self.pixel_size = self.map_size / self.pixel_number
        else:
            raise missing_variable_error("map variables")
        
        self.pixel_number = int(self.map_size/self.pixel_size)
        x, y = np.linspace(-self.map_size/2, self.map_size/2, self.pixel_number), np.linspace(-self.map_size/2, self.map_size/2, self.pixel_number)
        # arcmin grid
        self.X_arcmin, self.Y_arcmin = np.meshgrid(x, y)
        # degree grid
        self.X_deg, self.Y_deg = self.X_arcmin/60, self.Y_arcmin/60
        self.map = coord.SkyCoord(ra=self.X_deg * u.degree, 
                               dec=self.Y_deg * u.degree)
        
        return None
    
    # def make_map(self, pixel_size: float = None, pixel_number: int = None) -> None:
    #     if pixel_number:
    #         self.pixel_number = pixel_number
    #     if pixel_size:
    #         self.pixel_size = pixel_size
    #     if self.pixel_number is None or self.pixel_size is None:
    #         raise ValueError("pixel_number and pixel_size must be set")

    #     # integer-centered coordinates
    #     x, y, = (np.arange(self.pixel_number) - self.pixel_number // 2) * self.pixel_size

    #     X, Y = np.meshgrid(x, y, indexing="xy")

    #     self.X = X
    #     self.Y = Y
    #     self.R = np.sqrt(X**2 + Y**2)

    #     self.R_check = True
    #     print("Real-space coordinates made (FFT-safe).")
    #     return None
    
    def _calc_angular_seperation(self, map_size: float = None, pixel_size: float = None, pixel_number: int = None) -> None:
        if map_size or pixel_size or pixel_number:
            self.make_map(map_size, pixel_size, pixel_number)
        self.ang_sep = self.map.separation(self.cluster_coords).value * (np.pi / 180 )
        return None
    
    def _calc_ref_density(self, rho_type: str = None, z: float = None) -> None:
        if z:
            self.z = z
        if rho_type:
            self.rho_type = rho_type
        if self.rho_type == 'crit':
            print(self.cosmology)
            rho_c_z = self.cosmology.critical_density(self.z)
        elif self.rho_type == 'mean':
            print(self.cosmology)
            rho_c_z = self.cosmology.critical_density(self.z) * self.cosmology.critical_density(self.z)
        else:
            raise ValueError("rho_def must be 'crit' or 'mean'")
        
        self.rho_c_z = rho_c_z.to('M_sun/Mpc3')
        return None
    
    def _calc_virial_r(self, cluster_mass: float = None, delta: float = None, ref_dens: float = None, rho_type: str = None, z: float = None) -> None:
        if cluster_mass:
            self.M = cluster_mass * u.Msun
        if delta:
            self.delta = delta
        if ref_dens:
            self.rho_c_z = ref_dens
        #if rho_type or z or (not self.rho_c_z):
        self._calc_ref_density(rho_type, z)
        self.r_v = ((self.M / (self.delta * 4. * np.pi / 3.) / self.rho_c_z)**(1./3.)).to('Mpc')
        return None
    
    def _calc_concentration(self, cluster_mass: float = None, delta: float = None, ref_dens: float = None, rho_type: str = None, z: float = None) -> None:
        if cluster_mass:
            self.M = cluster_mass * u.Msun
        if delta:
            self.delta = delta
        if ref_dens:
            self.rho_c_z = ref_dens
        #if rho_type or z or (not self.rho_c_z):
        self._calc_ref_density(rho_type, z)
        self.concentration = halo_concentration.concentration(self.M.value, '%s%s' % (self.delta, self.rho_type[0]), self.z)
        return None

    def _calc_nfwConcentration(self, cluster_mass: float = None, delta: float = None, ref_dens: float = None, rho_type: str = None, z: float = None) -> None:
        if delta:
            self.delta = delta
        if any([cluster_mass, delta, ref_dens, rho_type, z]) or (not self.concentration):
            self._calc_concentration(cluster_mass, delta, ref_dens, rho_type, z)
        self.nfwConcentration = (self.delta / 3.) * (self.concentration**3.) / (np.log(1. + self.concentration) - self.concentration / (1. + self.concentration))
        return None
    
    def _calc_nfwScaleRadius(self, cluster_mass: float = None, delta: float = None, ref_dens: float = None, rho_type: str = None, z: float = None) -> None:
        #if any([cluster_mass, delta, ref_dens, rho_type, z]) or (not self.r_v) or (not self.concentration):
        self._calc_virial_r(cluster_mass, delta, ref_dens, rho_type, z)
        self._calc_concentration(cluster_mass, delta, ref_dens, rho_type, z)
        self.r_s = self.r_v.to('Mpc') / self.concentration
        return None
    
    def _calc_lensing_distances(self, z: float = None, z_source: float = None) -> None:
        if z:
            self.z = z
        if z_source:
            self.z_source = z_source
        self.D_l = self.cosmology.comoving_distance(self.z) / (1. + self.z)  # Lens distance
        self.D_s = self.cosmology.comoving_distance(self.z_source) / (1. + self.z_source)  # Source distance
        self.D_ls = (self.cosmology.comoving_distance(self.z_source) - self.cosmology.comoving_distance(self.z)) / (1. + self.z_source)
        return None
    
    def _calc_crit_surface_density(self, z: float = None, z_source: float = None) -> None:
        if any([z, z_source]) or (not any([self.D_l, self.D_s, self.D_ls])):
            self._calc_lensing_distances(z, z_source)
        self.sigma_c = (((const.c.cgs**2.) / (4. * np.pi * const.G.cgs)) * 
               (self.D_s / (self.D_l * self.D_ls))).to('M_sun/Mpc2')
        return None
    
    # def _calc_crit_surface_density(self, z: float = None, z_source: float = None, G:float = None, c: float = None) -> None:
    #     if G:
    #            self.G = G
    #     if c:
    #            self.c = c
        #   if any([z, z_source]):
        #     self._calc_lensing_distances(z, z_source)
    #     self.sigma_c = (((const.c.cgs**2.) / (4. * np.pi * const.G.cgs)) * 
    #            (self.D_s / (self.D_l * self.D_ls))).to('M_sun/Mpc2')
    #     return None

    def _calc_pyhsical_dist(self, map_size: float = None, pixel_size: float = None, pixel_number: int = None, z: float = None, z_source: float = None, cluster_mass: float = None, delta: float = None, ref_dens: float = None, rho_type: str = None) -> None:
        if any([map_size, pixel_size, pixel_number]) or (not self.ang_sep):
            self._calc_angular_seperation(map_size, pixel_size, pixel_number)
        #if any([z, z_source]) or (not any([self.D_l, self.D_s, self.D_ls])):
        self._calc_lensing_distances(z, z_source)
        if any([z, cluster_mass, delta, ref_dens, rho_type]) or (not self.r_s):
            self._calc_nfwScaleRadius(cluster_mass, delta, ref_dens, rho_type, z)
        physical_r = self.D_l * self.ang_sep  # Physical radius in Mpc
        #self.physical_r = physical_r * u.Mpc
        self.physical_r_scaled = physical_r / self.r_s
        self.physical_r_scaled = self.physical_r_scaled.value
        #physical_r_scaled = self.physical_r / self.r_s  # Dimensionless radius
        #self.physical_r_scaled = physical_r_scaled.decompose().value
        return None
    
    def _calc_nfwProfile(self, map_size: float = None, pixel_size: float = None, pixel_number: int = None, z: float = None, z_source: float = None, cluster_mass: float = None, delta: float = None, ref_dens: float = None, rho_type: str = None) -> None:
        if any([map_size, pixel_size, pixel_number, z, z_source, cluster_mass, delta, ref_dens, rho_type]) or (not self.physical_r_scaled):
            self._calc_pyhsical_dist(map_size, pixel_size, pixel_number, z, z_source, cluster_mass, delta, ref_dens, rho_type)
        proj = np.zeros(self.physical_r_scaled.shape)
        proj[np.where(self.physical_r_scaled > 1.0)] = (1. / (self.physical_r_scaled[np.where(self.physical_r_scaled > 1.0)]**2. - 1)) * \
                      (1. - (2. / np.sqrt(self.physical_r_scaled[np.where(self.physical_r_scaled > 1.0)]**2. - 1.)) * 
                       np.arctan(np.sqrt((self.physical_r_scaled[np.where(self.physical_r_scaled > 1.0)] - 1.) / (self.physical_r_scaled[np.where(self.physical_r_scaled > 1.0)] + 1.))))
        proj[np.where((self.physical_r_scaled >= self.physical_r_scaledMin) & (self.physical_r_scaled < 1.0))] = (1. / (self.physical_r_scaled[np.where((self.physical_r_scaled >= self.physical_r_scaledMin) & (self.physical_r_scaled < 1.0))]**2. - 1)) * \
                           (1. - (2. / np.sqrt(1. - self.physical_r_scaled[np.where((self.physical_r_scaled >= self.physical_r_scaledMin) & (self.physical_r_scaled < 1.0))]**2.)) * 
                            np.arctanh(np.sqrt((1. - self.physical_r_scaled[np.where((self.physical_r_scaled >= self.physical_r_scaledMin) & (self.physical_r_scaled < 1.0))]) / (self.physical_r_scaled[np.where((self.physical_r_scaled >= self.physical_r_scaledMin) & (self.physical_r_scaled < 1.0))] + 1.))))
        self.center_sing_extension = np.where((self.physical_r_scaled < self.physical_r_scaledMin) & (self.physical_r_scaled < 1.0))
        if len(self.center_sing_extension[0]) > 0:
            physical_r_small = self.physical_r_scaled[self.center_sing_extension]
            proj[self.center_sing_extension] = 1./3. + (2./15.) * physical_r_small**2
        analytic_limit = np.where(np.abs(self.physical_r_scaled - 1.0) < 1.0e-5)
        proj[analytic_limit] = 1. / 3.
        self.proj = proj
        return None
    
    def proj_surf_density(self, map_size: float = None, pixel_size: float = None, pixel_number: int = None, z: float = None, z_source: float = None, cluster_mass: float = None, delta: float = None, ref_dens: float = None, rho_type: str = None) -> None:
        #if any([map_size, pixel_size, pixel_number, z, z_source, cluster_mass, delta, ref_dens, rho_type]) or (not self.proj.any()) or (not self.nfwConcentration):
        self._calc_nfwProfile(map_size, pixel_size, pixel_number, z, z_source, cluster_mass, delta, ref_dens, rho_type)
        self._calc_nfwScaleRadius(cluster_mass, delta, ref_dens, rho_type, z)
        self._calc_nfwConcentration(cluster_mass, delta, ref_dens, rho_type, z)
        self._calc_ref_density(rho_type, z)
        self.sigma = ((2. * self.r_s * self.nfwConcentration * self.rho_c_z) * self.proj).to('M_sun/Mpc2')
        return None
    
    def kappa(self, map_size: float = None, pixel_size: float = None, pixel_number: int = None, z: float = None, z_source: float = None, cluster_mass: float = None, delta: float = None, ref_dens: float = None, rho_type: str = None) -> None:
        #if any([map_size, pixel_size, pixel_number, z, z_source, cluster_mass, delta, ref_dens, rho_type]) or (not self.sigma):
        self._calc_crit_surface_density(z, z_source)
        self.proj_surf_density(map_size, pixel_size, pixel_number, z, z_source, cluster_mass, delta, ref_dens, rho_type)
        self.Kappa = (self.sigma / self.sigma_c).value
        return None
    
    def _make_multipole_magnitude_2d(self, map_size: float = None, pixel_size: float = None):
        if map_size:
            self.map_size = map_size
        if pixel_size:
            self.pixel_size = pixel_size
        self.pixel_number = int(self.map_size/self.pixel_size)
        # Convert to radians for Fourier space
        dx_rad, dy_rad = (np.pi/180)*(self.pixel_size/60.), (np.pi/180)*(self.pixel_size/60.)
        x, y = np.fft.fftfreq(self.pixel_number, dx_rad), np.fft.fftfreq(self.pixel_number, dy_rad)
        # Convert to angular frequency (2π factor)
        x, y = 2*np.pi*x, 2*np.pi*y
        self.lX, self.lY = np.meshgrid(x, y)
        self.l2d = np.sqrt(self.lX**2 + self.lY**2)
        self.l2d_check = True
        return None
    
    def kappa2alpha(self) -> None:
        if not self.l2d_check:
            self._make_multipole_magnitude_2d()
        kappa_fft = np.fft.fft2(self.Kappa)
        phi_fft = -2. * kappa_fft / (self.l2d**2 + + 1e-30)

        # Deflection angle: α̃ = -i ℓ φ̃ (gradient in Fourier space)
        self.alphaX_fft = -1j * self.lX * phi_fft
        self.alphaY_fft = -1j * self.lY * phi_fft
        
        # Handle division by zero at ℓ=0
        self.alphaX_fft[np.isnan(self.alphaX_fft)] = 0
        self.alphaY_fft[np.isnan(self.alphaY_fft)] = 0
        
        # Inverse FFT to get real-space deflection angles
        # Convert from radians to arcmin
        self.alphaX = np.degrees(np.fft.ifft2(self.alphaX_fft).real) * 60
        self.alphaY = np.degrees(np.fft.ifft2(self.alphaY_fft).real) * 60
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

    def _calc_hubble_parameter(self, h_zero: float = None, omega_m: float = None, omega_l: float = None, z: float = None) -> float:
        if h_zero:
            self.h_zero = h_zero/100
            print("h_zero changed to: " + str(h_zero))
        if any([omega_m, omega_l, z]):
            self._calc_hubble_parameter_factor(omega_m, omega_l, z)

        self.hubble_parameter = np.sqrt(self.h_zero**2 * self.hubble_parameter_factor)
        print("hubble parameter = " + str(self.hubble_parameter))
        return self.hubble_parameter


class lens_profile():
    """ 
        class is mostly reused/cleaned code from Awais Mirza 2019 at Argelander Institut Bonn

        lens is presumed to be NFW
    """


    def __init__(self, extent = 3, n = 100, cluster_mass = 5e14, cluster_redshift = 0.7, r_max = 10, hubble_constant = 67.74, cosmology = "planck18", omega_m = 0.3089, omega_l = 0.6911, grav_constant = 4.30091 * 10**(-3) / (10**(6)), speed_of_light = scipy.constants.speed_of_light):
        ## cluster parameters
        self.M = cluster_mass  ## in M_sun
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

        self.n = n

        self.pos_map = None
        self.theta_max = extent
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

        self.set_angle_map(extent, int(self.n))

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
        self.theta_max = theta_max
        self.n = int(n)

        dec = np.linspace(-theta_max, theta_max, int(n))    #[arcmin]
        ra = np.linspace(theta_max, -theta_max, int(n))     #[arcmin]


        pos_map = np.meshgrid(dec, ra) # [arcmin]

        self.pos_map = np.array(pos_map)
        print('pos map set!')
        print("##############################################")
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
        #d_theta = 0.54 * (self.v_max)**2 * (self.comoving_dist - self._calc_comoving_dist(z=1100.0)) * (self._calc_f(angles=theta) / self._calc_comoving_dist(z = 1100.0))
        self.d_theta = 0.54 * (self.v_max)**2 * (self.comoving_dist - self._calc_comoving_dist(z=1100.0)) * (self.f / self._calc_comoving_dist(z = 1100.0))
        print("angular displacement calculated!")
        return None
    
    def plot_d_theta(self) -> None:
        if not self.d_theta.any():
            raise missing_internal_variable_error("d_theta")
        im = plt.imshow(-1*self.d_theta * 1e7, origin='lower', extent=[0, self.theta_max, 0, self.theta_max], interpolation='bilinear', cmap='RdBu')
        im.set_clim()

        plt.xlabel(r"$\Delta\theta_x$[arcmin]")
        plt.ylabel(r"$\Delta\theta_y$[arcmin]")

        cbar = plt.colorbar(im)
        cbar.set_label(r"$\alpha$ [arcmin]")   # label for the scale
        return None
    
    def lensing_map(self) -> list[list[float]]:
        x = np.linspace(-self.theta_max/2, self.theta_max/2, self.n)
        y = np.linspace(-self.theta_max/2, self.theta_max/2, self.n)
        x_mesh, y_mesh = np.meshgrid(x, y)


        theta = np.sqrt(x_mesh**2 + y_mesh**2)

        d_theta_x = (-self.d_theta * x_mesh/theta) * 1e6
        d_theta_y = (-self.d_theta * y_mesh/theta) * 1e6
        print("##############################################")
        print("Map of lensing angles created!")
        print("##############################################" + "\n" + "Done!")
        return d_theta_x, d_theta_y