"""
Nonlinear PSF correction from summary parameters only (no PSF image).

The output is built as a spin-2 combination

    z_out = sum_k c_k(I) z_k  +  c_p(I) z_p

where z_k are the K spin-2 fields the head already produces, z_p = e_psf, and
every coefficient is an arbitrary nonlinear function of an invariant vector I.
This is exactly the structure the group permits: spin-2 basis vectors, invariant
coefficients. Nothing else is allowed, and nothing else is needed.

What goes into I:

  - the attention-pooled invariant features q of the reconstruction,
  - the conditioning embedding c built from (log_snr, rho, log_T_psf, |e_psf|),
  - |z_p|, and the magnitudes |z_k|,
  - the contractions Re(z_k conj(z_p)) = |z_k||z_p| cos(2 dphi).

That last block is the point. It is the relative position angle between the
galaxy's own shape and the beam, and it is what an additive correction cannot
see. Note that Im(z_k conj(z_p))^2 = |z_k|^2|z_p|^2 - Re(...)^2 is already
implied by the terms above, so the parity-even pairwise geometry is complete
and no reflection-odd quantity ever enters. Equivariance is exact.

Why this should beat  e = NN(recon) - a * e_psf  specifically:

  For Gaussians the quadrupoles just add and the additive form is right to
  leading order with a = T_psf/(T_gal+T_psf). The residual on SKA-MID was not
  that coefficient -- it was that HQS-PnP and the deblender do not act linearly
  on the PSF. Their nonlinearity is governed by how much of the reconstruction
  came from the data rather than from DRUNet's prior, which is precisely what
  log_snr and rho measure. So c_p is made a function of them, along with the
  galaxy-beam orientation and the galaxy's own recovered morphology.

Initialisation and transfer:

  The coefficient MLP is zero-initialised, giving c_k = 1 and c_p = 0, and the
  pooling is algebraically identical to Spin2Head's. Module names are unchanged,
  so load_true_weights() gives a model whose forward pass is bit-for-bit the
  pretrained true-image network. The correction is learned strictly as a
  departure from an m ~ 1e-4 solution.

Conditioning vector, standardised by the dataloader:

    cond[:, 0:3] = (log_snr, rho, log_T_psf)
    cond[:, 3:5] = (e1_psf, e2_psf) / s
"""

import torch
from escnn import nn

from .shape_network_v2 import EqBackbone, Spin2Head, bound_ellipticity

__all__ = ["ShapeNet_PSFCorr", "COND_LAYOUT", "check_equivariance_corr"]

COND_LAYOUT = ("log_snr", "rho", "log_T_psf", "e1_psf", "e2_psf")

EPS = 1e-12


def _zero_(layer):
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    return layer


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
class CondMLP(torch.nn.Module):
    """Invariant scalars -> shared conditioning embedding."""

    def __init__(self, n_in, hidden=64, cond_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, cond_dim),
            torch.nn.SiLU(),
        )

    def forward(self, s):
        return self.net(s)


class FieldFiLM(torch.nn.Module):
    """
    Per-field multiplicative gain on a GeometricTensor.

    One scalar per field, broadcast across that field's components, so the gain
    commutes with the group action irrespective of the field's representation
    or change of basis.
    """

    def __init__(self, ftype, cond_dim):
        super().__init__()
        self.ftype = ftype
        sizes = torch.tensor([r.size for r in ftype.representations])
        self.register_buffer("sizes", sizes, persistent=False)
        self.gamma = _zero_(torch.nn.Linear(cond_dim, len(ftype.representations)))

    def forward(self, x, c):
        g = 1.0 + self.gamma(c)
        g = torch.repeat_interleave(g, self.sizes, dim=1)
        return nn.GeometricTensor(x.tensor * g[:, :, None, None], x.type)


# ---------------------------------------------------------------------------
# Backbone and head
# ---------------------------------------------------------------------------
class EqBackboneFiLM(EqBackbone):
    """EqBackbone with a gain FiLM after each block."""

    def __init__(self, cond_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.film1 = FieldFiLM(self.block1.out_type, cond_dim)
        self.film2 = FieldFiLM(self.block2.out_type, cond_dim)
        self.film3 = FieldFiLM(self.block3.out_type, cond_dim)

    def forward(self, image, c):
        x = self.input_type(self.norm(image))
        x = self.mask(x)
        x = self.film1(self.block1(x), c)
        x = self.film2(self.block2(x), c)
        x = self.film3(self.block3(x), c)
        return x


class Spin2HeadNL(Spin2Head):
    """
    Spin2Head, but the K spin-2 fields are pooled *before* being summed, so a
    nonlinear invariant function can weight them and mix in the PSF direction.

    Reuses to_spin / to_inv / mix unchanged, so the pretrained head transfers.
    """

    def __init__(
        self,
        r2_act,
        feat_type,
        K=16,
        C=32,
        hidden=64,
        cond_dim=64,
        coef_hidden=128,
    ):
        super().__init__(r2_act, feat_type, K=K, C=C, hidden=hidden)

        self.film_inv = _zero_(torch.nn.Linear(cond_dim, 2 * C))

        n_in = C + cond_dim + 1 + 2 * K  # q, c, |z_p|, |z_k|, Re(z_k conj z_p)
        self.coef = torch.nn.Sequential(
            torch.nn.BatchNorm1d(n_in),
            torch.nn.Linear(n_in, coef_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(coef_hidden, coef_hidden),
            torch.nn.SiLU(),
            _zero_(torch.nn.Linear(coef_hidden, K + 1)),
        )

    def forward(self, f, c, z_p):
        B = f.tensor.shape[0]

        v = self.to_spin(f).tensor  # (B, 2K, H, W)
        s = self.to_inv(f).tensor  # (B,  C, H, W)

        gamma, beta = self.film_inv(c).chunk(2, dim=-1)
        s = s * (1.0 + gamma)[:, :, None, None] + beta[:, :, None, None]

        h = self.mix(s)
        w, a = h[:, : self.K], h[:, self.K :]

        H, W = v.shape[-2:]
        v = v.view(B, self.K, 2, H, W)
        a = torch.softmax(a.flatten(2), dim=-1).view(B, 1, H, W)

        # pooled spin-2 vectors and pooled invariant features
        z = (w.unsqueeze(2) * v * a.unsqueeze(1)).sum(dim=(-2, -1))  # (B, K, 2)
        q = (s * a).sum(dim=(-2, -1))  # (B, C)

        # invariants: magnitudes and the galaxy-beam contraction
        z_norm = torch.sqrt((z * z).sum(-1) + EPS)  # (B, K)
        zp_norm = torch.sqrt((z_p * z_p).sum(-1, keepdim=True) + EPS)  # (B, 1)
        re = (z * z_p[:, None, :]).sum(-1)  # (B, K)

        coef = self.coef(torch.cat([q, c, zp_norm, z_norm, re], dim=-1))
        c_k = 1.0 + coef[:, : self.K]
        c_p = coef[:, self.K :]

        return (c_k.unsqueeze(-1) * z).sum(dim=1) + c_p * z_p


class ShapeNet_PSFCorr(torch.nn.Module):
    def __init__(
        self,
        size=128,
        K=16,
        C=32,
        hidden=64,
        dropout=0.1,
        stride3=1,
        flip_e2=False,
        n_scalars=3,
        cond_dim=64,
        cond_hidden=64,
        coef_hidden=128,
        use_e_norm=True,
    ):
        super().__init__()
        self.n_scalars = n_scalars
        self.use_e_norm = use_e_norm

        self.cond = CondMLP(
            n_scalars + int(use_e_norm), hidden=cond_hidden, cond_dim=cond_dim
        )
        self.eq = EqBackboneFiLM(
            cond_dim=cond_dim, size=size, dropout=dropout, stride3=stride3
        )
        self.head = Spin2HeadNL(
            self.eq.r2_act,
            self.eq.out_type,
            K=K,
            C=C,
            hidden=hidden,
            cond_dim=cond_dim,
            coef_hidden=coef_hidden,
        )
        self.register_buffer(
            "_sign", torch.tensor([1.0, -1.0 if flip_e2 else 1.0]), persistent=False
        )

    def forward(self, im, cond):
        n = self.n_scalars
        scalars = cond[:, :n]
        z_p = cond[:, n : n + 2] * self._sign  # data convention -> internal basis

        if self.use_e_norm:
            scalars = torch.cat([scalars, z_p.norm(dim=1, keepdim=True)], dim=1)

        c = self.cond(scalars)
        e = self.head(self.eq(im, c), c, z_p)
        return bound_ellipticity(e * self._sign)

    def load_true_weights(self, path, map_location="cpu"):
        """Every pretrained tensor transfers; the new modules are zero-init, so
        the model starts as an exact copy of the true-image network."""
        sd = torch.load(path, map_location=map_location)
        return self.load_state_dict(sd, strict=False)


@torch.no_grad()
def check_equivariance_corr(model, x, cond, verbose=True):
    """
    e_psf is spin-2 and must be transformed with the image: rot90 sends it to
    -e_psf, a horizontal flip sends it to (e1, -e2). Residuals should sit at
    float noise. Call with model.eval().
    """
    was_training = model.training
    model.eval()

    n = model.n_scalars
    sign = torch.tensor([1.0, -1.0], device=x.device, dtype=x.dtype)

    cond_rot, cond_flip = cond.clone(), cond.clone()
    cond_rot[:, n : n + 2] *= -1.0
    cond_flip[:, n : n + 2] *= sign

    e = model(x, cond)
    e_rot = model(torch.rot90(x, 1, dims=(-2, -1)), cond_rot)
    e_flip = model(torch.flip(x, dims=(-1,)), cond_flip)

    err_rot = (e_rot + e).abs().max().item()
    err_flip = (e_flip - e * sign).abs().max().item()

    if verbose:
        print(f"rot90 residual : {err_rot:.3e}")
        print(f"flip  residual : {err_flip:.3e}")
        if err_flip > 1e-3 and err_rot < 1e-4:
            print("  -> flip fails but rotation passes: try flip_e2=True")

    if was_training:
        model.train()
    return err_rot, err_flip


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ShapeNet_PSFCorr(size=128, flip_e2=True)
    x = torch.randn(4, 1, 128, 128)
    cond = torch.randn(4, 5)
    print("output shape:", model(x, cond).shape)
    check_equivariance_corr(model, x, cond)
