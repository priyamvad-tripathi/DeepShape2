"""
FiLM-conditioned equivariant shape measurement network.

Extends ShapeNet_v2 with two conditioning pathways that respect O(2):

  invariant scalars  (log_snr, rho, log_T_psf)  -> FiLM
      Scalars are invariant under the group, so they may modulate anything.
      They enter as a *per-field gain* on the equivariant backbone features
      and as full gain+shift on the invariant branch of the head.

  spin-2 pair        (e1_psf, e2_psf)           -> explicit spin-2 field
      This is NOT invariant: it rotates as exp(2i*theta) and conjugates under
      reflection. Feeding it through FiLM would inject a fixed direction in the
      ellipticity plane and destroy equivariance. Instead it is appended as an
      extra spin-2 channel alongside the K learned ones in the head, with an
      invariant, image-dependent gate. The network can therefore learn
      e_out = e_meas - a(snr, rho, T) * e_psf, which is exactly the PSF
      anisotropy correction, without breaking the transformation law.

Why the backbone FiLM is gain-only:
  Multiplying a whole field by a scalar commutes with any linear group action,
  so a gain is equivariant in any basis. A shift is not: adding a constant to
  the components of a non-trivial irrep breaks the transformation law (this is
  why escnn's R2Conv only puts bias on trivial irreps). Shifts are applied only
  on the head's invariant branch, where the fields are genuinely trivial.

Initialisation:
  Every conditioning output layer is zero-initialised, so at init
      ShapeNet_FiLM(im, cond) == ShapeNet_v2(im)
  exactly. Fine-tuning therefore starts at the pretrained function and the
  conditioning is learned as a departure from it. Module names are unchanged
  (eq.block*, head.to_spin, head.to_inv, head.mix), so
      model.load_state_dict(torch.load("shapenet_true_g.pt"), strict=False)
  transfers the whole pretrained model and leaves only the new modules random.

Expected conditioning vector (already standardised by the dataloader):

      cond[:, 0:3] = (log_snr, rho, log_T_psf)     invariant, z-scored
      cond[:, 3:5] = (e1_psf, e2_psf) / s          spin-2, isotropic scale only

  |e_psf| is not passed: it is the norm of the spin-2 pair, so the model forms
  it internally and appends it to the invariant block for free.
"""

import torch
from escnn import nn

# Adjust to wherever ShapeNet_v2 lives in deepshape2.models
from deepshape2.models.shape_network_v2 import EqBackbone, Spin2Head, bound_ellipticity

__all__ = ["ShapeNet_FiLM", "COND_LAYOUT", "check_equivariance_film"]

COND_LAYOUT = ("log_snr", "rho", "log_T_psf", "e1_psf", "e2_psf")


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


class Spin2HeadFiLM(Spin2Head):
    """
    Spin2Head plus:
      - gain/shift FiLM on the invariant branch (trivial fields, so legal),
      - an extra spin-2 channel carrying e_psf, gated by an invariant,
        image-dependent weight map.

    `mix` keeps its original output width (K gates + 1 pooling logit) so the
    pretrained head transfers unchanged; the PSF gate is a separate branch.
    """

    def __init__(self, r2_act, feat_type, K=16, C=32, hidden=64, cond_dim=64):
        super().__init__(r2_act, feat_type, K=K, C=C, hidden=hidden)

        self.film_inv = _zero_(torch.nn.Linear(cond_dim, 2 * C))

        self.psf_gate = torch.nn.Sequential(
            torch.nn.BatchNorm2d(C),
            torch.nn.SiLU(),
            torch.nn.Conv2d(C, hidden, 1),
            torch.nn.SiLU(),
            _zero_(torch.nn.Conv2d(hidden, 1, 1)),
        )

    def forward(self, f, c, e_psf):
        B = f.tensor.shape[0]

        v = self.to_spin(f).tensor  # (B, 2K, H, W)
        s = self.to_inv(f).tensor  # (B,  C, H, W)

        gamma, beta = self.film_inv(c).chunk(2, dim=-1)
        s = s * (1.0 + gamma)[:, :, None, None] + beta[:, :, None, None]

        h = self.mix(s)
        w, a = h[:, : self.K], h[:, self.K :]

        H, W = v.shape[-2:]
        v = v.view(B, self.K, 2, H, W)

        # learned spin-2 fields + the PSF spin-2 vector, invariantly gated
        e_map = (w.unsqueeze(2) * v).sum(dim=1)
        e_map = e_map + self.psf_gate(s) * e_psf[:, :, None, None]

        a = torch.softmax(a.flatten(2), dim=-1).view(B, 1, H, W)
        return (e_map * a).sum(dim=(-2, -1))  # (B, 2)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class ShapeNet_FiLM(torch.nn.Module):
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
        self.head = Spin2HeadFiLM(
            self.eq.r2_act,
            self.eq.out_type,
            K=K,
            C=C,
            hidden=hidden,
            cond_dim=cond_dim,
        )
        self.register_buffer(
            "_sign", torch.tensor([1.0, -1.0 if flip_e2 else 1.0]), persistent=False
        )

    def forward(self, im, cond):
        n = self.n_scalars
        scalars = cond[:, :n]
        e_psf = cond[:, n : n + 2] * self._sign  # data convention -> internal basis

        if self.use_e_norm:
            scalars = torch.cat([scalars, e_psf.norm(dim=1, keepdim=True)], dim=1)

        c = self.cond(scalars)
        e = self.head(self.eq(im, c), c, e_psf)
        return bound_ellipticity(e * self._sign)

    def load_true_weights(self, path, map_location="cpu"):
        """Load shapenet_true_g.pt; new conditioning modules stay at their
        zero-init, so the model starts as an exact copy of the true-image net."""
        sd = torch.load(path, map_location=map_location)
        return self.load_state_dict(sd, strict=False)


# ---------------------------------------------------------------------------
# Equivariance test
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_equivariance_film(model, x, cond, verbose=True):
    """
    The PSF ellipticity is spin-2, so it must be transformed alongside the
    image: rot90 sends e -> -e (2*theta = 180 deg) and a horizontal flip sends
    e -> (e1, -e2). Both the input e_psf and the output must follow that law.

    Call with model.eval(); BatchNorm in train mode will break this.
    """
    was_training = model.training
    model.eval()

    n = model.n_scalars
    sign = torch.tensor([1.0, -1.0], device=x.device, dtype=x.dtype)

    cond_rot = cond.clone()
    cond_rot[:, n : n + 2] *= -1.0

    cond_flip = cond.clone()
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
    model = ShapeNet_FiLM(size=128, flip_e2=True)
    x = torch.randn(4, 1, 128, 128)
    cond = torch.randn(4, 5)
    print("output shape:", model(x, cond).shape)
    check_equivariance_film(model, x, cond)
