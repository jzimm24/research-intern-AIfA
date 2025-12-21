Code for producing lensing distortions on continuum maps.
 
This code was developed as a research internship at AIfa at the group of Prof. Dr. Frank Bertoldi
under the supervision of Dr. Kaustuv Moni Basu.
 
The code can be divided into three sections, each corresponding to its own class.
For an example use case please take a look at the notebook "cmb_lensing_example.ipynb"
 
map_maker class:
 
Class for producing maps of the Cosmic Microwave Background.
The map parameters (map size, pixel number and pixel size) can be set as well as the spectrum type
and spectrum parameters depending on the spectrum type (for accurrate CMB maps take spectrum_type="acoustic").
Additionally the amount of multipoles in Fourier space (l_degrees) should be set.
Currently only squared map types are supported.
The class outputs the created gaussian random field (CMB) in both real- and Fourier-space as well as its RMS.

Example use:

map = map_maker(map_size=100, pixel_size=0.2, l_degrees=6000)


nfw_lens class:

Class for producing deflection angle maps representing the impact of an NFW galaxy cluster on the position of a background source.
The map parameters (map size, pixel number and pixel size) can be set, but should match the map parameters of the map_maker class
if the maps produced by this class are the ones to be ultimatly lensed.
Additionally the clusters redshift and mass can be defined and the following parameters might be changed to account for different
cosmologies: type of reference density, Hubbel constant, omega_m, gravitational constant, speed of light, delta, source redshift.
The primary output of the class is a map of deflection angles. This means an array of size (2, map_size, map_size) constaining the lensing in both dimensions.
This map can also be accessed in Fourier space. The deflection angles are computed by taking the space derivative of the convergence in Fourier space.

Example use: 

lens = nfw_lens()
lens.make_map(map_extension, resolution)
lens.kappa()
lens.kappa2alpha()

lensing class:

Class for producing a post lensing map given a scalar map (CMB) and a deflection angle map matching in map parameters.
The map parameters can be passed at initialization (should match parameters of both other maps) or be copied from the pre-lensing map.
The class computes the post-lensing map by pushing the value from the pre-lensing map at each position by the values given by the deflection angle map at this position.

Example use:

output = lensing(pre_lens_map=background_map_real, lens = alpha2)
output.lensing_map = alpha2

output.copy_map_parameters()
output._make_pos_map()

output.apply_lensing()
