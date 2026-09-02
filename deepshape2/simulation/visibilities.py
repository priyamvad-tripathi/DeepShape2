# %% Imports
import gc

import numpy as np
import xarray
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.image import Image
from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.visibility import create_visibility, export_visibility_to_ms
from ska_sdp_func_python.imaging import (
    create_image_from_visibility,
    invert_ng,
    predict_ng,
)
from ska_sdp_func_python.imaging.weighting import (
    taper_visibility_gaussian,
    weight_visibility,
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
    "apply_uv_cut",
    "_image_to_rascil",
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
    Convert a sky image to a RASCIL Image with WCS and polarisation frame.

    Parameters
    ----------
    image_array : ndarray
        Either (ny, nx) for a single channel, or (nchan, ny, nx) for a cube
        already sampled at `frequency`.
    phasecentre : SkyCoord
        Phase centre; becomes CRVAL1/2.

    Keyword Arguments
    -----------------
    cellsize : float
        Pixel size in radians.
    frequency : float or ndarray
        Channel centre frequencies in Hz. Only the first `nchan` are used.
    channel_bandwidth : float or ndarray
        Channel widths in Hz. Sets the FREQ cdelt when nchan == 1; for
        nchan > 1 the cdelt is the channel spacing taken from `frequency`.
    nchan : int or None
        Number of channels in the output Image. None (default) means "match
        the input array", so a 2D input gives a single-channel Image. Setting
        nchan > 1 with a 2D input replicates the plane across channels.
    spectral_index : float
        Only used when replicating a 2D input across nchan > 1. Flux is scaled
        by (nu / nu_ref) ** spectral_index with nu_ref = frequency[0].
        Default 0.0, i.e. a flat spectrum.

    Returns
    -------
    Image

    """
    cellsize = kwargs.get("cellsize", CELLSIZE)
    frequency = kwargs.get("frequency", FREQUENCY)
    channel_bandwidth = kwargs.get("channel_bandwidth", BANDWIDTH)
    nchan = kwargs.get("nchan", None)
    spectral_index = kwargs.get("spectral_index", 0.0)

    frequency = np.atleast_1d(np.asarray(frequency, dtype=float))
    channel_bandwidth = np.atleast_1d(np.asarray(channel_bandwidth, dtype=float))

    arr = np.asarray(image_array)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim != 3:
        raise ValueError(
            f"image_array must be (ny, nx) or (nchan, ny, nx), got {arr.shape}"
        )

    n_in, ny, nx = arr.shape
    if nchan is None:
        nchan = n_in

    if nchan > len(frequency):
        raise ValueError(
            f"nchan={nchan} but only {len(frequency)} frequencies supplied"
        )

    if n_in == nchan:
        image = arr.astype(np.float64, copy=True)
    elif n_in == 1:
        # replicate the single plane, optionally with a power-law spectrum
        scale = (frequency[:nchan] / frequency[0]) ** float(spectral_index)
        image = arr.astype(np.float64, copy=True) * scale[:, None, None]
    else:
        raise ValueError(
            f"cannot map {n_in} input channels onto nchan={nchan}; supply "
            f"either 1 plane or exactly {nchan}"
        )

    np.nan_to_num(image, copy=False)
    image = image.reshape(nchan, 1, ny, nx)  # (nchan, npol, ny, nx)

    # FREQ axis: cdelt is the channel spacing when there is more than one
    # channel, and falls back to the channel width for a single channel.
    if nchan > 1:
        dfreq = float(frequency[1] - frequency[0])
        spacing = np.diff(frequency[:nchan])
        if not np.allclose(spacing, dfreq, rtol=1e-6):
            raise ValueError(
                "frequency axis must be evenly spaced for a WCS FREQ axis; "
                f"got spacings {spacing}"
            )
    else:
        dfreq = float(channel_bandwidth[0])

    cellsize_deg = cellsize * 180.0 / np.pi

    w = WCS(naxis=4)
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 0, float(frequency[0])]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "STOKES", "FREQ"]
    w.wcs.cdelt = [-cellsize_deg, +cellsize_deg, 1, dfreq]
    w.wcs.radesys = "ICRS"
    w.wcs.equinox = 2000.0
    w.wcs.crpix = [nx // 2 + 1, ny // 2 + 1, 1, 1]  # WCS order is (x, y, ...)

    return Image.constructor(
        image,
        wcs=w,
        polarisation_frame=PolarisationFrame("stokesI"),
        clean_beam=None,
    )


def predict_visibilities_from_array(image_array, ra_deg, dec_deg, **kwargs):
    """
    Predict visibilities from a sky image.

    Parameters
    ----------
    image_array : ndarray
        (ny, nx) or (nchan, ny, nx) sky image in Jy/pixel.
    ra_deg, dec_deg : float
        Phase centre in degrees.

    Keyword Arguments
    -----------------
    nchan : int or None
        Channels in the model Image. None (default) matches the input array,
        so a 2D image gives a single-channel model and predict_ng maps every
        visibility channel onto it (flat spectrum).
    spectral_index : float
        Power-law index used when replicating a 2D image across nchan > 1.
    vis : xarray.Dataset or None
        Predict onto this template instead of building a new one. Use it when
        the weights or uv cut have already been set up.
    Remaining kwargs are forwarded to _image_to_rascil and
    create_visibility_template (cellsize, frequency, channel_bandwidth,
    ha_interval, integration_time, tel, rmax, ...).

    Returns
    -------
    xarray.Dataset
    """
    verbosity = kwargs.get("verbosity", 0)
    threads = kwargs.get("threads", 20)
    vis = kwargs.pop("vis", None)

    phasecentre = SkyCoord(
        ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs", equinox="J2000"
    )

    im = _image_to_rascil(image_array, phasecentre, **kwargs)

    if vis is None:
        vis = create_visibility_template(phasecentre, **kwargs)

    return predict_ng(vis, im, verbosity=verbosity, threads=threads)


def make_dirty_image_and_psf(vis: xarray.Dataset, **kwargs):
    """
    Create a dirty image and optionally a PSF from visibilities.

    The Briggs uv cell is 1/(npixel * cellsize), so the grid used to count
    the uv density fixes the weighting, while the grid used for the inversion
    only fixes the output image size. NPIX_grid lets the two differ: weight on
    a large grid matching the reference imaging (e.g. WSClean's -size), then
    invert onto a small grid holding just the PSF. The large model is freed
    before the inversion, so it never coexists with the gridder's internals.

    Parameters
    ----------
    vis : xarray.Dataset
            Visibility dataset.
    NPIX : int
            Number of pixels along each axis of the OUTPUT image.
    NPIX_grid : int or None
            Number of pixels along each axis of the grid used to compute the
            imaging weights. If None (default) or equal to NPIX, a single
            model is used for both weighting and inversion.
    cellsize : float
            Pixel size in radians. Used for both grids, so NPIX_grid also
            sets the weighting field of view.
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
    taper: float
            Tapering parameter for imaging (in arcsec).
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
    NPIX_grid = kwargs.get("NPIX_grid", None)
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
    taper = kwargs.get("taper", None)

    decouple = NPIX_grid is not None and NPIX_grid != NPIX

    # Grid used to count the uv density -> sets the Briggs weights.
    model_w = create_image_from_visibility(
        vis,
        cellsize=cellsize,
        npixel=NPIX_grid if decouple else NPIX,
        override_cellsize=override_cellsize,
    )

    vis_reweighted = weight_visibility(
        vis, model=model_w, weighting=weighting, robustness=robustness
    )

    if decouple:
        # Free the large weighting grid before the gridder allocates.
        del model_w
        gc.collect()
        model = create_image_from_visibility(
            vis, cellsize=cellsize, npixel=NPIX, override_cellsize=override_cellsize
        )
    else:
        model = model_w

    if taper is not None:
        ARCSEC = np.pi / (180 * 3600)
        vis_reweighted = taper_visibility_gaussian(vis_reweighted, beam=taper * ARCSEC)

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
    # noise = np.vectorize(complex)(noise_real, noise_imag)
    noise = noise_real + 1j * noise_imag

    noisy_vis = vis.copy(deep=False)
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

    newvis = vis.copy(deep=False)

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


# %%
def apply_uv_cut(vis: xarray.Dataset, **kwargs):
    """
    Zero the imaging weights of visibilities outside a uv-distance range.

    Mirrors WSClean's -minuv-l / -maxuv-l. Must be applied BEFORE Briggs
    weighting, so the cut samples never enter the uv density counts.

    Parameters
    ----------
    vis : xarray.Dataset
            Visibility dataset to cut.
    minuv_l : float or None
            Lower uv-distance limit in wavelengths (default 80.0).
    maxuv_l : float or None
            Upper uv-distance limit in wavelengths (default None, no cut).
    return_frac : bool
            If True, return the fraction of samples cut alongside the dataset.
    verbosity : int
            If >0, print the cut fraction.
    verify : bool
            If True, re-read through flagged_imaging_weight and raise if any
            weight inside the cut survived.

    Returns
    -------
    cut_vis : xarray.Dataset
            Copy of vis with the out-of-range weights zeroed.
    frac : float (optional)
            Fraction of samples cut (returned only if return_frac is True).
    """

    minuv_l = kwargs.get("minuv_l", 80.0)
    maxuv_l = kwargs.get("maxuv_l", None)
    return_frac = kwargs.get("return_frac", False)
    verbosity = kwargs.get("verbosity", 0)
    verify = kwargs.get("verify", True)

    uvw = vis.visibility_acc.uvw_lambda
    uvd = np.hypot(uvw[..., 0], uvw[..., 1])

    mask = np.zeros(uvd.shape, dtype=bool)
    if minuv_l is not None:
        mask |= uvd < minuv_l
    if maxuv_l is not None:
        mask |= uvd > maxuv_l

    cut_vis = vis.copy(deep=False)
    for name in ("imaging_weight", "weight"):
        if name in cut_vis:
            cut_vis[name].data[mask] = 0.0

    frac = float(mask.mean())

    if verbosity > 0:
        print(
            f"uv cut [{minuv_l}, {maxuv_l}] lambda: {frac * 100:0.2f}% of "
            f"samples zeroed"
        )

    if verify:
        left = cut_vis.visibility_acc.flagged_imaging_weight[mask]
        if left.size and np.abs(left).max() > 0:
            raise RuntimeError(
                f"uv cut did not take: max weight inside the cut is "
                f"{np.abs(left).max():.3e}. Check which data_var holds the "
                f"imaging weights on this Visibility."
            )

    if return_frac:
        return cut_vis, frac
    return cut_vis
