# %% Import Libraries and Modules
import sys

sys.path.append("/home/tripathi/Scripts/")
sys.path.append("/home/tripathi/DeepShape2/")


import helpers
import numpy as np
import simulation as sim

NPIX_mosaic = 256

# %% Simulate wide-field sky
patch, centre, bottom_left = sim.random_patch()
patch = sim.filter_patch_by_size(sim.filter_patch_by_flux(patch))

sky, patch_out, isolated_stamps_all = sim.simulate_wide_field(
    patch, bottom_left, verbosity=1
)

# Generate the visibilities
vt = sim.simulate_visibilities_ms(
    field=sky,
    pointing=centre,
    filename="/scratch/tripathi/DS2/Trial/sky.ms",
)

# %% Create stamps
locations = np.array([patch_out["pix_x"], patch_out["pix_y"]]).T

# peak = np.array([np.max(iso) for iso in isolated_stamps_all])

mask = patch_out["flux"].values > 50e-6

# Extract isolated stamps
isolated_stamps_large = helpers.get_stamps(
    isolated_stamps_all, NPIX=int(NPIX_mosaic * 2)
)
isolated_stamps = helpers.extract_image(isolated_stamps_large[mask], NPIX=NPIX_mosaic)

#  Extract blended stamps
blended_stamps_large = helpers.extract_multiple(
    sky, centers=locations[mask], NPIX=int(NPIX_mosaic * 2), relative=False
)
blended_stamps = helpers.extract_image(blended_stamps_large, NPIX=NPIX_mosaic)
