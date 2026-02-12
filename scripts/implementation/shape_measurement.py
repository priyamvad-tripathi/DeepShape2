# %%
import numpy as np

from deepshape2.utils import blendedness, correlations, load_config, load_h5
from deepshape2.visualization import (
    contour_plot,
    plot_bias,
    plot_residual_slope,
    shape_error_vs_flux,
)

# %% Load Data

cfg = load_config()
DATA_DIR = cfg["DATA_DIR"]
HF_PATH = DATA_DIR + "deep_set.h5"
RESULTS_DIR = cfg["RESULTS_DIR"]


hf = load_h5(HF_PATH, "r")["patch_000"]

df = hf["patch_df"][()]
flux_mask = df["flux_mask"]

shapes = np.stack(
    [df["e1"][flux_mask], df["e2"][flux_mask]],
    axis=1,
).astype(np.float32)

flux = df["flux"][flux_mask].astype(np.float32)
peaks = hf["peaks"][:]


ypred_wf = hf["shape_pred"][:]
ypred_iso = hf["isolated_dirty_psf"]["shape_pred"][:]


# %%
def evaluate_predictions(ypred, shapes, label, mask=None, power=1e3, lim=0.4):
    if mask is None:
        mask = np.ones(len(shapes), dtype=bool)

    corr0 = np.corrcoef(ypred[mask, 0], shapes[mask, 0])[0, 1]
    corr1 = np.corrcoef(ypred[mask, 1], shapes[mask, 1])[0, 1]
    print(f"{label} : {corr0:.3f}/{corr1:.3f}")

    shape_norm = np.linalg.norm(ypred[mask] - shapes[mask], axis=1)
    p25, median, p75 = np.percentile(shape_norm, [25, 50, 75])
    print(f"Shape norm: {median:.2f} + {p75 - median:.2f} - {median - p25:.2f}")

    plot_bias(
        ypred,
        shapes,
        power=power,
        lim=lim,
        ellipticity_cutoff=1,
        bad_index=~mask,
    )


# %% Individual plots

mask = (flux > 50 * 1e-6) & (peaks > 3 * 0.71e-6)
evaluate_predictions(ypred_wf, shapes, "WF (filtered)", mask=mask)
print("-----")
evaluate_predictions(ypred_iso, shapes, "Iso (filtered)", mask=mask)

# %% Correlations
isolated_stamps = hf["isolated_stamps"][:]
blended_stamps = hf["blended_stamps"][:]
flux_mask = df["flux_mask"]
param_dict = {
    "flux": df["flux"][flux_mask] * 1e6,
    "size": df["size"][flux_mask],
    "peak": np.max(isolated_stamps, axis=(1, 2)),
    "blendedness": blendedness(isolated_stamps, blended_stamps),
    "sersic": df["sersic_index"][flux_mask],
    "peak_dirty": peaks,
}
shape_diff_iso = np.linalg.norm(ypred_iso - shapes, axis=1)
shape_diff_wf = np.linalg.norm(ypred_wf - shapes, axis=1)
correlations(
    param_dict, [shape_diff_iso, shape_diff_wf], ["Shape err Iso", "Shape err WF"]
)

# %% Plotting
# contour_plot(
#     shapes[:, 0],
#     [ypred_iso[:, 0], ypred_wf[:, 0]],
#     legends=["Isolated", "Wide field"],
#     lim=0.52,
#     pow=1e3,
#     fname=RESULTS_DIR + "shape_measurement/contour_ds.pdf",
# )

contour_plot(
    shapes[mask, 0],
    [ypred_iso[mask, 0], ypred_wf[mask, 0]],
    legends=["Isolated", "Wide field"],
    lim=0.36,
    pow=1e3,
    fname=RESULTS_DIR + "shape_measurement/contour_ds_filtered.pdf",
)

# %%

factor = (np.max(hf["dirty"][mask], axis=(1, 2)) - 0.71e-06) / peaks[mask]

bins = np.array([factor.min(), 1, 1.25, 1.5, 2, factor.max()])
bin_centers = [0.85, 1.12, 1.35, 1.75, 2.25]


plot_residual_slope(
    factor,
    ypred_wf[mask],
    shapes[mask],
    bins=bins,
    bin_centers=bin_centers,
    fname=RESULTS_DIR + "shape_measurement/residual_slope_wf.pdf",
)

# %%
shape_error_vs_flux(
    [shape_diff_iso[mask], shape_diff_wf[mask]],
    flux[mask] * 1e6,
    bin_edges=[50, 60, 100, 150, 200],
    # fname=RESULTS_DIR + "shape_measurement/shape_error_vs_flux.pdf",
)
