from matplotlib.pyplot import legend
from normalization import *
import numpy as np
import pickle as pkl

filepath = '/home/li/Documents/sss-correspondence/data/GullmarsfjordSMaRC20210209_ssh_annotations/survey2_better_resolution/9-0169to0182-nbr_pings-1301_annotated/patch240_step40_test0.1_refSSH-0170/test/pkl/patch50_SSH-0170_pings_1180to1420_bins_0to1301_isport_False.pkl'
with open(filepath, 'rb') as f:
    data = pkl.load(f)
raw_intensities = data.sss_waterfall_image[100, :]

frequency = 400 * 10**3
wavelength = 1500 / frequency
mounting_angle = math.radians(30)
vertical_opening = math.radians(70)
transducer_radius = estimate_transducer_radius(wavelength, vertical_opening)
h = 20  # m

print(raw_intensities.shape)
slant_ranges = np.arange(0, 200, 200/raw_intensities.shape[0])
print(slant_ranges)

modelled_intensities = np.zeros_like(slant_ranges)
corrected_intensities = np.zeros_like(slant_ranges)

for i in range(slant_ranges.shape[0]):
    hit = SSSHit(r_s=slant_ranges[i], h=h)
    try:
        modelled_intensities[i] = modelled_echo_intensity(hit, frequency, wavelength,
                                                mounting_angle,
                                                transducer_radius)
        corrected_intensities[i] = intensity_correction(hit, raw_intensities[i], modelled_intensities[i])
    except ValueError as e:
        print(f'Cannot compute asin for {hit}')
                                            
# normalize intensities to (0, 1)
raw_intensities_sum = raw_intensities.sum()
raw_intensities_norm = (raw_intensities - raw_intensities.min())/(raw_intensities.max() - raw_intensities.min())
modelled_intensities_norm = (modelled_intensities - modelled_intensities.min())/(modelled_intensities.max() - modelled_intensities.min())

corrected_intensities[:150] = 0
corrected_intensities/=raw_intensities_sum
#corrected_intensities = (corrected_intensities - corrected_intensities.min())/(corrected_intensities.max() - corrected_intensities.min())
#corrected_intensities*= raw_intensities
corrected_intensities = (corrected_intensities - corrected_intensities.min())/(corrected_intensities.max() - corrected_intensities.min())


plt.plot(raw_intensities_norm, 'b', label='raw')
plt.plot(modelled_intensities_norm, 'y', label='modelled')
plt.plot(corrected_intensities, 'g', label='corrected')
plt.show()