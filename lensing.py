import numpy as np
import matplotlib.pyplot as plt
import scipy

class lensing():
    def __init__(self, map_size: float = None, pixel_size: float = None, pixel_number: int = None, dec_shift = 0, asc_shift = 0, lens: list[list[float]] = None, pre_lens_map: list[list[float]] = None, poly_degree = 2):
        self.map_size = map_size
        self.pixel_size = pixel_size
        self.pixel_number = pixel_number
        self.dec_shift = dec_shift
        self.asc_shift = asc_shift
        self.pre_lens_map = pre_lens_map
        self.l_total = poly_degree
        ## internal vatiables
        self.pos_map = None
        self.pos_map_fs = None
        self.lensing_map = lens
        ## output
        self.post_lensing_map = None

    
    def set_NFW_variables(self) -> None:
        return None
    
    def make_NFW_profile(self) -> None:
        return None
    
    def set_map_parameters(self, map_size: float = None, pixel_size: float = None, pixel_number: int = None) -> None:
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
    
    def make_lensing_map():
        return None
    
    def set_lensing_map(self, map) -> list[list[float]]:
        self.lensing_map = map
        return None
    
    def set_pre_lens_map(self, map: list[list[float]]) -> None:
        self.pre_lens_map = map
        return None
    
    def copy_map_parameters(self):
        if not (self.pre_lens_map).any():
            print("!!!!!!!!!!!!!!!!! Please first provide a pre lens map.")
        elif ((self.pre_lens_map).shape[0] != (self.pre_lens_map).shape[1]):
            print("!!!!!!!!!!Please provide squared map.")
        else:
            self.pixel_number = (self.pre_lens_map).shape[0]
            self.pixel_size = 1                                     ######### Temporary solution
            self.map_size = self.pixel_number                       ######### Temporary solution
            print("Map parameters: map_size = " + str(self.map_size) + ", pixel_size = " + str(self.pixel_size) + ", pixel_number = " + str(self.pixel_number) + ".")
        return None
    
    # def _make_background_map(self) -> list[list[list[float]], float]:
    #     mean = np.mean(self.pre_lens_map)
    #     bg_map = np.ones((self.pixel_number, self.pixel_number))*mean
    #     self.background_map = bg_map
    #     return bg_map, mean
    
    def _make_pos_map(self, shift_dec = 0, shift_asc = 0):
        # real_space
        dec, asc = np.linspace(-self.map_size/2, self.map_size/2, self.pixel_number), np.linspace(-self.map_size/2, self.map_size/2, self.pixel_number)
        pos_map = np.meshgrid(dec, asc)
        ## Necessary because tuples can only be read (not modified)
        pos_map_obj_1, pos_map_obj_2 = pos_map 
        pos_map_obj_1 += shift_dec
        pos_map_obj_2 += shift_asc
        self.pos_map =  pos_map_obj_1, pos_map_obj_2

        # fourier_space
        x, y = 2*np.pi*np.fft.fftfreq(self.pixel_number, (np.pi*180)*self.pixel_size/60), 2*np.pi*np.fft.fftfreq(self.pixel_number, (np.pi*180)*self.pixel_size/60)  ## pixel_size given in seconds
        self.pos_map_fs = np.meshgrid(x, y)
        return None
    
    # def _make_pos_map_test(self, shift_dec = 0, shift_asc = 0) -> None:
    #     # real_space
    #     dec, asc = np.linspace(-self.map_size/2, self.map_size/2, self.pixel_number), np.linspace(-self.map_size/2, self.map_size/2, self.pixel_number)
    #     pos_map = np.meshgrid(dec, asc)
    #     ## Necessary because tuples can only be read (not modified)
    #     pos_map_obj_1, pos_map_obj_2 = pos_map 
    #     pos_map_obj_1 += shift_dec
    #     pos_map_obj_2 += shift_asc
    #     self.pos_map =  pos_map_obj_1, pos_map_obj_2

    #     # fourier_space
    #     self.pos_map_fs = np.fft.fft2(self.pos_map)

    #     return None
    
    def apply_lensing(self, pre_lens_map: list[list[float]]= None, lens: list[list[float]] = None, shift_dec = 0, shift_asc = 0):
        if pre_lens_map:
            self.pre_lens_map = pre_lens_map

        if lens:
            self.lensing_map = lens

        self._make_pos_map(shift_dec, shift_asc)
        dec_pos, asc_pos = self.pos_map
        post_lensing_dec_pos, post_lensing_asc_pos = dec_pos + self.lensing_map[0], asc_pos + self.lensing_map[1]
        #interpolate = scipy.interpolate.RectBivariateSpline(asc_pos[:,0], dec_pos[0,:], self.pre_lens_map, kx = self.l_total, ky = self.l_total)
        #interpolate = scipy.interpolate.RectBivariateSpline(dec_pos[:,0], asc_pos[0,:], self.pre_lens_map, kx = self.l_total, ky = self.l_total)
        interpolate = scipy.interpolate.RectBivariateSpline(asc_pos[:,0], dec_pos[0,:], self.pre_lens_map, kx = self.l_total, ky = self.l_total)
        self.post_lensing_map  = interpolate.ev(post_lensing_asc_pos.flatten(), post_lensing_dec_pos.flatten()).reshape([len(asc_pos), len(dec_pos)]) 

        return None
    
    def get_lensed_map(self) -> list[list[float]]:
        return self.post_lensing_map
    
    def plot_lensed_map():
        return None