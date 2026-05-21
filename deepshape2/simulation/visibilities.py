# %% Imports
import numpy as np
import xarray
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.gridded_visibility import create_griddata_from_image
from ska_sdp_datamodels.image import Image
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.visibility import create_visibility, export_visibility_to_ms
from ska_sdp_func_python.grid_data import (
    grid_visibility_weight_to_griddata,
    griddata_visibility_reweight,
)
from ska_sdp_func_python.imaging import (
    create_image_from_visibility,
    invert_ng,
    predict_ng,
)
from ska_sdp_func_python.util.coordinate_support import skycoord_to_lmn
from ska_sdp_func_python.visibility import (
    calculate_visibility_phasor,
    calculate_visibility_uvw_lambda,
    phaserotate_visibility,
)

from ..utils.io import load_config

__all__ = [
    "create_visibility_template",
    "make_dirty_image_and_psf",
    "add_noise_to_visibility",
    "predict_visibilities_from_array",
    "rephase_visibility",
    "simulate_visibilities",
]

# %% Load default configuration
cfg = load_config()

FREQUENCY = cfg["frequency"]
BANDWIDTH = cfg.get("bandwidth", 0.3 * FREQUENCY)
INTEGRATION_TIME = cfg["integration_time"]
HA_INTERVAL = cfg["ha_interval"]
CELLSIZE = cfg["SCALE_RADIANS"]

SCALE_RADIANS = cfg["SCALE_RADIANS"]

NPIX_SKY = cfg["NPIX_SKY"]


# %% Core helpers


def create_visibility_template(phasecentre, tel="MID", rmax=None, **kwargs):
    """
    Create an empty visibility dataset for a telescope configuration.

    Parameters
    ----------
    phasecentre : SkyCoord
            Phase centre of the observation.
    tel : str
            Named telescope configuration (default "MID").
    rmax : float or None
            Maximum baseline length in meters.
    frequency : ndarray or None
            Frequency array in Hz. Defaults to FREQUENCY.
    channel_bandwidth : ndarray or None
            Channel bandwidths in Hz. Defaults to BANDWIDTH.
    ha_interval : tuple or None
            Hour-angle interval in hours. Defaults to HA_INTERVAL.
    integration_time : float or None
            Integration time in seconds. Defaults to INTEGRATION_TIME.

    Returns
    -------
    xarray.Dataset
            Visibility dataset ready for prediction.
    """
    if not isinstance(phasecentre, SkyCoord):
        raise TypeError(
            "phasecentre should be an astropy.coordinates.SkyCoord instance"
        )

    frequency = kwargs.get("frequency", FREQUENCY)
    channel_bandwidth = kwargs.get("channel_bandwidth", BANDWIDTH)
    ha_interval = kwargs.get("ha_interval", HA_INTERVAL)
    integration_time = kwargs.get("integration_time", INTEGRATION_TIME)

    if isinstance(frequency, (float, int)):
        frequency = np.array([frequency])
    if isinstance(channel_bandwidth, (float, int)):
        channel_bandwidth = np.array([channel_bandwidth])

    config = create_named_configuration(tel, rmax=rmax)

    # Number of integration samples across the hour angle range
    dtime_hr = integration_time / 3600.0
    ntimes = int((ha_interval[1] - ha_interval[0]) / dtime_hr)

    # Times centered with respect to transit (convert hours to radians)
    times = (
        np.linspace(
            ha_interval[0] + dtime_hr / 2.0,
            ha_interval[1] - dtime_hr / 2.0,
            ntimes,
        )
        * np.pi
        / 12.0
    )

    vt = create_visibility(
        config,
        times,
        frequency,
        channel_bandwidth=channel_bandwidth,
        weight=1.0,
        phasecentre=phasecentre,
        polarisation_frame=PolarisationFrame("stokesI"),
        elevation_limit=None,
    )

    return vt


def _image_to_rascil(image_array, phasecentre, **kwargs):
    """
    Helper: convert a 2D image array to a RASCIL Image with WCS and polarisation.
    Expects 'cellsize', 'frequency', and 'channel_bandwidth' in kwargs.
    """
    cellsize = kwargs.get("cellsize", CELLSIZE)
    frequency = kwargs.get("frequency", FREQUENCY)
    channel_bandwidth = kwargs.get("channel_bandwidth", BANDWIDTH)

    if not isinstance(frequency, float):
        frequency = float(frequency[0])

    ny, nx = image_array.shape
    image = image_array.reshape([1, 1, ny, nx])
    np.nan_to_num(image, copy=False)

    cellsize_deg = cellsize * 180.0 / np.pi

    w = WCS(naxis=4)
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 0, frequency]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    w.wcs.cdelt = [-cellsize_deg, +cellsize_deg, 1, channel_bandwidth]
    w.wcs.radesys = "ICRS"
    w.wcs.equinox = 2000.0
    w.wcs.crpix = [ny // 2 + 1, nx // 2 + 1, 1, 1]

    polarisation_frame = PolarisationFrame("stokesI")
    return Image.constructor(
        image, wcs=w, polarisation_frame=polarisation_frame, clean_beam=None
    )


def predict_visibilities_from_array(image_array, ra_deg, dec_deg, **kwargs):
    """
    Predict visibilities from a 2D image array.

    Parameters
    ----------
    image_array : ndarray
        2D sky image (ny, nx).
    ra_deg, dec_deg : float
        Phase centre coordinates in degrees.
    kwargs : dict
        Additional parameters passed to create_visibility_set or predict_ng.

    Returns
    -------
    xarray.Dataset
        Predicted visibilities.
    """

    verbosity = kwargs.get("verbosity", 0)
    threads = kwargs.get("threads", 20)

    phasecentre = SkyCoord(
        ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs", equinox="J2000"
    )

    # Predict visibilities onto provided visibility template
    im = _image_to_rascil(image_array, phasecentre, **kwargs)

    visibility_template = create_visibility_template(phasecentre, **kwargs)

    vt = predict_ng(visibility_template, im, verbosity=verbosity, threads=threads)
    return vt


def make_dirty_image_and_psf(vis: xarray.Dataset, **kwargs):
    """
    Create a dirty image and optionally a PSF from visibilities.

    Parameters
    ----------
    vis : xarray.Dataset
            Visibility dataset.
    NPIX : int
            Number of pixels along each axis.
    cellsize : float
            Pixel size in radians.
    weighting : str
            'natural', 'uniform', or 'robust'.
    robustness : float
            Robustness parameter used for weighting.
    override_cellsize : bool
            Pass through to create_image_from_visibility.
    verbosity : int
            Verbosity level.
    do_wstacking : bool
            Use W-stacking in the imaging.
    do_psf : bool
            Also compute the PSF.
    asarray : bool
            Return numpy arrays instead of Image objects.

    Returns
    -------
    dirty, psf : ndarray or Image
            Dirty image (and PSF if requested).
    """

    NPIX = kwargs.get("NPIX", 128)
    cellsize = kwargs.get("cellsize", CELLSIZE)
    weighting = kwargs.get("weighting", "robust")
    robustness = kwargs.get("robustness", -0.5)
    override_cellsize = kwargs.get("override_cellsize", False)
    verbosity = kwargs.get("verbosity", 0)
    do_wstacking = kwargs.get("do_wstacking", True)
    do_psf = kwargs.get("do_psf", True)
    do_dirty = kwargs.get("do_dirty", True)
    asarray = kwargs.get("asarray", True)
    threads = kwargs.get("threads", 20)

    model = create_image_from_visibility(
        vis, cellsize=cellsize, npixel=NPIX, override_cellsize=override_cellsize
    )

    # Compute grid weights and reweight visibilities
    grid_weights = create_griddata_from_image(
        model, polarisation_frame=model.image_acc.polarisation_frame
    )
    grid_weights = grid_visibility_weight_to_griddata(vis, grid_weights)
    vis_reweighted = griddata_visibility_reweight(
        vis,
        grid_weights[0],
        weighting=weighting,
        robustness=robustness,
        sumwt=grid_weights[1],
    )

    # Invert to obtain dirty image
    dirty_img, psf_img = None, None
    if do_dirty:
        dirty_img, _ = invert_ng(
            vis_reweighted,
            model,
            verbosity=verbosity,
            do_wstacking=do_wstacking,
            threads=threads,
        )

    if do_psf:
        psf_img, _ = invert_ng(
            vis_reweighted,
            model,
            dopsf=True,
            verbosity=verbosity,
            do_wstacking=do_wstacking,
            threads=threads,
        )

    def _to_array(img):
        return img.pixels.to_numpy().astype(np.float32).squeeze()

    if asarray:
        if do_dirty and do_psf:
            return _to_array(dirty_img), _to_array(psf_img)
        if do_dirty:
            return _to_array(dirty_img)
        if do_psf:
            return _to_array(psf_img)

    if do_psf:
        return dirty_img, psf_img, _to_array(dirty_img), _to_array(psf_img)
    return dirty_img, _to_array(dirty_img)


def add_noise_to_visibility(vis: xarray.Dataset, **kwargs):
    """
    Add Gaussian noise to visibility data.

    Parameters
    ----------
    vis : xarray.Dataset
            Visibility dataset to which noise will be added.
    return_snr : bool
            If True, return computed SNR along with noisy visibilities.
    verbosity : int
            If >0, print RMS and SNR information.
    noise_fac : float or None
            Scale factor applied to the computed thermal noise.
    sigma : float or None
            If provided, set noise level explicitly in microJy.

    Returns
    -------
    noisy_vis : xarray.Dataset
            Copy of vis with noise added to vis.vis.
    snr : float (optional)
            Signal-to-noise ratio (returned only if return_snr is True).
    """

    return_snr = kwargs.get("return_snr", False)
    verbosity = kwargs.get("verbosity", 0)
    noise_fac = kwargs.get("noise_fac", None)
    sigma = kwargs.get("sigma", None)

    # Physical constants / telescope assumptions (SKA-MID based defaults)
    eta = 0.98  # Aperture efficiency
    k_b = 1.38064852e-23  # Boltzmann constant
    bandwidth = (vis.channel_bandwidth.data,)
    int_time = (vis["integration_time"].data,)
    sens = 10.1  # A_eff/T_sys
    bt = np.outer(int_time, bandwidth)
    sigma_arr = (np.sqrt(2) * k_b) / (sens * eta * (np.sqrt(bt)))
    sigma_val = sigma_arr[0, 0] * 1e26  # convert to Jy

    if noise_fac is not None:
        sigma_val = noise_fac * sigma_val

    if sigma is not None:
        # User-provided sigma assumed in microJy
        sigma_val = sigma * 1e-6

    # Compute SNR in visibility domain
    if sigma_val != 0:
        SNR = np.linalg.norm(vis.vis.data) / sigma_val
    else:
        SNR = np.inf

    if verbosity > 0:
        print(f"RMS Noise= {sigma_val * 1e6:0.2f} uJy")
        print(f"SNR_vis= {SNR:0.2f}")

    # Generate complex Gaussian noise (real and imaginary parts)
    noise_real = np.random.normal(loc=0.0, scale=sigma_val, size=vis.vis.shape)
    noise_imag = np.random.normal(loc=0.0, scale=sigma_val, size=vis.vis.shape)
    noise = np.vectorize(complex)(noise_real, noise_imag)

    noisy_vis = vis.copy(deep=True)
    noisy_vis["vis"].data = vis.vis.data.copy() + noise

    if return_snr:
        return noisy_vis, SNR
    return noisy_vis


def rephase_visibility(
    vis: xarray.Dataset,
    pix_loc,
    remove_w=True,
    no_correction=False,
    npix_sky=NPIX_SKY,
):
    """
    Change the phase centre of a visibility dataset by applying a phasor and
    optionally rotating the uvw coordinates.

    Parameters
    ----------
    vis : xarray.Dataset
            Input visibility dataset.
    new_phasecentre
            Desired new phase centre.
    remove_w : bool
            If True, set the w component to zero after rotation.

    Returns
    -------
    xarray.Dataset
            New visibility dataset with updated vis and uvw (and phasecentre attr).
    """

    newvis = vis.copy(deep=True)

    # --- Compute pixel offsets from image centre ---
    x_pix, y_pix = pix_loc
    dx = x_pix - npix_sky // 2.0
    dy = y_pix - npix_sky // 2.0

    # --- Compute new phasecentre SkyCoord ---
    pointing_centre = newvis.attrs["phasecentre"]

    offset_ra = -dx * SCALE_RADIANS * u.rad
    offset_dec = dy * SCALE_RADIANS * u.rad

    new_phasecentre_skycoord = pointing_centre.spherical_offsets_by(
        offset_ra, offset_dec
    )

    if no_correction:
        newvis = phaserotate_visibility(newvis, new_phasecentre_skycoord, tangent=False)
        return newvis

    dl, dm, _ = skycoord_to_lmn(new_phasecentre_skycoord, newvis.phasecentre)
    dn = np.sqrt(1.0 - dl**2 - dm**2)

    if np.abs(dn) < 1e-15:
        # No meaningful change
        return newvis

    # Apply phase rotation phasor (multiply by conjugate to shift phasecentre)
    phasor = calculate_visibility_phasor(new_phasecentre_skycoord, newvis)
    if newvis["vis"].data.shape != phasor.shape:
        raise ValueError("Visibility and phasor shapes do not match")
    newvis["vis"].data *= np.conj(phasor)

    uvw_old = newvis["uvw"].data.copy()
    u_new = uvw_old[..., 0] - dl * uvw_old[..., 2] / dn
    v_new = uvw_old[..., 1] - dm * uvw_old[..., 2] / dn
    uvw_new = np.zeros_like(uvw_old)
    uvw_new[..., 0] = u_new
    uvw_new[..., 1] = v_new
    if not remove_w:  # ? Apply additional w-term correction using wstacking
        uvw_new[..., 2] = uvw_old[..., 2]
    newvis["uvw"].data = uvw_new.copy()

    # Update phasecentre attribute and recompute lambda units
    newvis.attrs["phasecentre"] = new_phasecentre_skycoord
    newvis = calculate_visibility_uvw_lambda(newvis)

    return newvis


# %% High-level simulation helper


def simulate_visibilities(
    field, ra_pointing, dec_pointing, filename=None, create_dirty=False, **kwargs
):
    """
    High-level helper to simulate visibilities from a 2D image.

    Parameters
    ----------
    field : ndarray
            2D image array (ny, nx).
    ra_pointing, dec_pointing : float
            Pointing centre in degrees.
    filename : str or None
            If provided, export the noisy visibilities to a Measurement Set.
    kwargs : forwarded to lower-level functions (e.g. integration_time, frequencies)

    Returns
    -------
    xarray.Dataset or (xarray.Dataset, ndarray)
            Noisy (or clean) visibility dataset, optionally with dirty image.
    """
    # Generate visibilities from image
    vt = predict_visibilities_from_array(field, ra_pointing, dec_pointing, **kwargs)

    # Add noise default: use computed thermal noise
    vt_n = add_noise_to_visibility(vt, **kwargs)

    # Optionally save to Measurement Set
    if filename is not None:
        export_visibility_to_ms(filename, [vt_n])

    # Optionally create dirty image
    if create_dirty:
        dirty_arr = make_dirty_image_and_psf(
            vt_n, NPIX=field.shape[0], do_psf=False, **kwargs
        )
        return vt_n, dirty_arr

    return vt_n
