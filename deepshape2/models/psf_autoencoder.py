"""PSF autoencoder for DeepShape III.

Replacement for the 4-layer conv AE whose capacity was ~99.8% concentrated in two
FC layers (65536 <-> 1152), which let the decoder memorise a mean training PSF and
ignore the latent.  Changes, in rough order of importance:

  * Capacity moved into the conv trunk.  The FC bottleneck is ~0.3M params instead
    of 151M, so "reproduce the training mean" is no longer the cheap minimum.
  * Optional supervised head on the latent (e1, e2, log T, sidelobe depth).  This
    is the structural guarantee that the latent carries what ShapeNet's FiLM
    conditioning needs; reconstruction alone does not provide it.
  * Peak normalisation + asinh compression.  Data lives in (-inf, 1] with the peak
    pinned at exactly 1, and sidelobes at the 1e-2 level get real gradient weight.
  * Linear output head.  The (-inf, 1] range is a property of the normalisation,
    not something an activation should enforce -- a saturating head cannot emit the
    endpoints the normalisation guarantees are present.
  * GroupNorm, not BatchNorm.  No running statistics to invalidate when the domain
    shifts from SKA-MID to ILT.
  * Bilinear upsample + conv instead of ConvTranspose2d.  Checkerboard artefacts at
    the 1e-3 level would sit exactly on top of the sidelobe structure that carries
    the discriminative content.
  * Exact 180-degree rotation symmetry enforced on the output.  For Hermitian uv
    coverage the dirty beam satisfies PSF(-x) = PSF(x) identically, so this is a
    free hard constraint, not a prior.

Latent is a plain vector with no output activation, sized for FiLM conditioning.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = [
#     "AsinhScale",
#     "PSFAutoencoder",
#     "peak_normalise",
#     "psf_loss",
#     "latent_health",
#     "latent_probe",
# ]


# --------------------------------------------------------------------------- #
# data transforms
# --------------------------------------------------------------------------- #


def peak_normalise(x, eps=1e-12):
    """Divide by the per-image peak.  Returns (normalised, peak).

    Peak -> exactly 1, sidelobes keep their sign, range is (-inf, 1].  Unlike
    per-image min-max this does not tie the dynamic range to the deepest sidelobe
    pixel, which is itself one of the quantities that varies across the facet
    family (and between SKA-MID and ILT).
    """
    peak = x.amax(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return x / peak, peak


class AsinhScale(nn.Module):
    """Signed log-like compression that fixes t(1) = 1 and t(0) = 0.

    t(x) = asinh(x / s) / asinh(1 / s)

    Preserves the (-inf, 1] range of peak-normalised data while giving a -5%
    sidelobe (s = 0.01) a value of -0.44 instead of -0.05, i.e. ~9x the gradient
    weight under MSE.  Plain MSE on linear peak-normalised PSFs is dominated by a
    handful of core pixels and the sidelobe structure is effectively unsupervised.
    """

    def __init__(self, softening=0.01):
        super().__init__()
        self.softening = float(softening)
        self.register_buffer(
            "norm", torch.tensor(math.asinh(1.0 / self.softening)), persistent=False
        )

    def forward(self, x):
        return torch.asinh(x / self.softening) / self.norm

    def inverse(self, y):
        return self.softening * torch.sinh(y * self.norm)


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #


def _groups(channels, target=8):
    g = min(target, channels)
    while channels % g:
        g -= 1
    return g


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(c_out), c_out)
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.op = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """Bilinear + conv.  ConvTranspose2d leaves checkerboard residue that lands in
    the same amplitude range as the sidelobes we are trying to encode."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


# --------------------------------------------------------------------------- #
# encoder / decoder
# --------------------------------------------------------------------------- #


class Encoder(nn.Module):
    def __init__(self, channels, blocks, latent_dim, bottleneck_ch, dropout=0.0):
        super().__init__()
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1)

        stages = []
        for i in range(len(channels)):
            c = channels[i]
            for _ in range(blocks[i]):
                stages.append(ResBlock(c, c, dropout=dropout))
            if i + 1 < len(channels):
                stages.append(Downsample(c, channels[i + 1]))
        self.stages = nn.Sequential(*stages)

        c_last = channels[-1]
        self.head = nn.Sequential(
            nn.GroupNorm(_groups(c_last), c_last),
            nn.SiLU(),
            nn.Conv2d(c_last, bottleneck_ch, 1),
        )
        self.spatial = 128 // 2 ** (len(channels) - 1)
        self.fc = nn.Linear(bottleneck_ch * self.spatial**2, latent_dim)

    def forward(self, x):
        h = self.stages(self.stem(x))
        h = self.head(h).flatten(1)
        return self.fc(h)  # no activation: a ReLU here kills half the latent


class Decoder(nn.Module):
    def __init__(self, channels, blocks, latent_dim, bottleneck_ch, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.bottleneck_ch = bottleneck_ch
        self.spatial = 128 // 2 ** (len(channels) - 1)

        self.fc = nn.Linear(latent_dim, bottleneck_ch * self.spatial**2)
        self.proj = nn.Conv2d(bottleneck_ch, channels[-1], 1)

        stages = []
        for i in reversed(range(len(channels))):
            c = channels[i]
            for _ in range(blocks[i]):
                stages.append(ResBlock(c, c, dropout=dropout))
            if i > 0:
                stages.append(Upsample(c, channels[i - 1]))
        self.stages = nn.Sequential(*stages)

        self.out_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], 1, 3, padding=1)
        # Down-scaled, not zeroed.  An exactly-zero output conv blocks gradient
        # into the entire decoder trunk on the first step (nothing flows back
        # through a zero weight), which is the same class of bug as a dead ReLU.
        # 0.1x keeps the initial output near flat without the dead step.
        with torch.no_grad():
            self.out_conv.weight.mul_(0.1)
            self.out_conv.bias.zero_()

    def forward(self, z):
        h = self.fc(z).view(-1, self.bottleneck_ch, self.spatial, self.spatial)
        h = self.stages(self.proj(h))
        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


class PSFAutoencoder(nn.Module):
    """Autoencoder over asinh-compressed, peak-normalised 128x128 PSF stamps.

    Parameters
    ----------
    latent_dim : int
        Size of the vector handed to ShapeNet's FiLM conditioning.  128 is ample:
        after factoring rotation out into ShapeNet's O(2) equivariance the ILT
        facet family is ~1-parameter (PC1 = 78.9%).  Keep it small -- an
        overparameterised latent is what let the old model leave it unused.
    n_aux : int
        Size of the supervised head on the latent.  Default 4 for
        (e1, e2, log T, min_sidelobe).  Set to 0 to disable.
    symmetrise : bool
        Enforce PSF(-x) = PSF(x) on the output.  Exact for Hermitian uv coverage.
    """

    def __init__(
        self,
        latent_dim=128,
        channels=(32, 64, 96, 128, 192),
        blocks=(1, 1, 2, 2, 2),
        bottleneck_ch=16,
        n_aux=4,
        softening=0.01,
        dropout=0.0,
        symmetrise=True,
        **kwargs,
    ):
        super().__init__()
        assert len(channels) == len(blocks)
        self.expected_image_shape = (1, 128, 128)
        self.latent_dim = latent_dim
        self.symmetrise = symmetrise

        self.scale = AsinhScale(softening)
        self.encoder = Encoder(channels, blocks, latent_dim, bottleneck_ch, dropout)
        self.decoder = Decoder(channels, blocks, latent_dim, bottleneck_ch, dropout)

        self.aux_head = (
            nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, n_aux))
            if n_aux
            else None
        )

    # -- core -------------------------------------------------------------- #

    def encode(self, x):
        """x: asinh-space stamps, (B, 1, 128, 128)."""
        return self.encoder(x)

    def decode(self, z):
        y = self.decoder(z)
        if self.symmetrise:
            y = 0.5 * (y + torch.flip(y, dims=(-2, -1)))
        return y

    def forward(self, x, **kwargs):
        z = self.encode(x)
        out = {"z": z, "recon": self.decode(z)}
        if self.aux_head is not None:
            out["aux"] = self.aux_head(z)
        return out

    # -- convenience ------------------------------------------------------- #

    @torch.no_grad()
    def embed(self, psf_raw, **kwargs):
        """Raw PSF stamps -> latent.  Applies peak normalisation and asinh."""
        x, _ = peak_normalise(psf_raw)
        return self.encode(self.scale(x))

    def to_linear(self, y):
        """asinh-space -> peak-normalised linear."""
        return self.scale.inverse(y)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #


def psf_loss(
    out,
    target,
    aux_target=None,
    w_aux=0.1,
    w_fourier=0.0,
    aux_weights=None,
    **kwargs,
):
    """Reconstruction (asinh space) + optional latent supervision + optional
    Fourier-magnitude term.

    `target` is the asinh-space stamp, i.e. the same thing fed to the encoder.
    `aux_target` is (B, n_aux), typically (e1, e2, log T, min_sidelobe) measured
    with gaussian_weighted_moments on the *linear* stamp.  Standardise these
    columns over the training set before use -- log T and e live on very different
    scales and an unstandardised aux loss will be dominated by whichever is larger.
    """
    recon = out["recon"]
    terms = {"recon": F.mse_loss(recon, target)}

    if w_fourier:
        # The uv-plane amplitude is the object the beam is a transform of;
        # supervising it directly stops the model trading sidelobe fidelity for
        # core sharpness.
        ft_r = torch.fft.rfft2(recon).abs()
        ft_t = torch.fft.rfft2(target).abs()
        terms["fourier"] = F.mse_loss(ft_r, ft_t) / (target.shape[-1] ** 2)

    if aux_target is not None and "aux" in out:
        se = (out["aux"] - aux_target) ** 2
        if aux_weights is not None:
            se = se * aux_weights
        terms["aux"] = se.mean()

    total = terms["recon"]
    total = total + w_fourier * terms.get("fourier", 0.0)
    total = total + w_aux * terms.get("aux", 0.0)
    return total, terms


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #


@torch.no_grad()
def latent_probe(z, y, names=None, frac_train=0.7, ridge=1e-3):
    """Can a linear map read the physical parameters back out of the latent?

    This, not reconstruction PSNR, is the selection metric.  The latent's job is
    to condition ShapeNet's FiLM on residual PSF structure, so what matters is
    whether it resolves the facet-to-facet differences -- for ILT, sigma(|e|) is
    0.014 across the 30 facets, so a probe residual well below that means the
    latent carries what is needed, and a residual comparable to it means the
    downstream conditioning cannot possibly work however good the reconstruction
    looks.

    Parameters
    ----------
    z : (N, D) latents.
    y : (N, K) physical targets, e.g. (e1, e2, log T, min_sidelobe).
    """
    z, y = z.float(), y.float()
    n = int(frac_train * len(z))
    zt, zv, yt, yv = z[:n], z[n:], y[:n], y[n:]

    mu, sd = zt.mean(0), zt.std(0).clamp_min(1e-8)
    zt, zv = (zt - mu) / sd, (zv - mu) / sd
    zt = torch.cat([zt, torch.ones(len(zt), 1)], 1)
    zv = torch.cat([zv, torch.ones(len(zv), 1)], 1)

    a = zt.T @ zt + ridge * len(zt) * torch.eye(zt.shape[1])
    w = torch.linalg.solve(a, zt.T @ yt)

    resid = zv @ w - yv
    rms = resid.pow(2).mean(0).sqrt()
    r2 = 1 - resid.pow(2).mean(0) / yv.var(0).clamp_min(1e-12)

    names = names or [f"y{i}" for i in range(y.shape[1])]
    return {
        n_: {"r2": float(a_), "rms": float(b_)} for n_, a_, b_ in zip(names, r2, rms)
    }


@torch.no_grad()
def latent_health(model, loader, device="cuda", max_batches=32):
    """Detect the failure mode of the old model: a latent the decoder ignores.

    Returns
    -------
    dict with
      active_dims  : dims whose std over the batch exceeds 1% of the mean std.
                     If this is a small fraction of latent_dim the bottleneck is
                     collapsing.
      eff_rank     : exp(entropy of the normalised PCA spectrum).  For a
                     1-parameter family plus learned rotation, ~3-8 is healthy;
                     ~1 means collapse, ~latent_dim means the latent is encoding
                     noise or nuisance structure.
      sensitivity  : mean |d recon| when the latent is perturbed by 1 sigma along
                     its leading PC, in asinh units.  Near zero => decoder is
                     running open-loop and reconstruction is a memorised prior.
    """
    model.eval().to(device)
    zs = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        zs.append(model.encode(x).float().cpu())
    z = torch.cat(zs)

    std = z.std(0)
    active = int((std > 0.01 * std.mean()).sum())

    zc = z - z.mean(0)
    s = torch.linalg.svdvals(zc) ** 2
    p = s / s.sum()
    eff_rank = float(torch.exp(-(p * (p + 1e-12).log()).sum()))

    _, _, v = torch.linalg.svd(zc, full_matrices=False)
    pc1 = v[0].to(device)
    z0 = z[:64].to(device)
    delta = float(
        (model.decode(z0 + std[0].item() * pc1) - model.decode(z0)).abs().mean()
    )

    return {"active_dims": active, "eff_rank": eff_rank, "sensitivity": delta}


if __name__ == "__main__":
    m = PSFAutoencoder()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 1, 128, 128)
    out = m(x)
    print(f"params      : {n / 1e6:.2f}M")
    print(f"recon shape : {tuple(out['recon'].shape)}")
    print(f"latent shape: {tuple(out['z'].shape)}")
    print(f"aux shape   : {tuple(out['aux'].shape)}")
