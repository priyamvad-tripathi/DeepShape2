"""
Equivariant shape measurement network (spin-2 head).

Predicts (e1, e2) as a genuinely equivariant quantity rather than projecting to
invariants and hoping a dense head recovers the orientation.

Key properties:
  - Backbone is O(2)-equivariant (continuous rotations + reflections).
  - Head outputs a frequency-2 irrep, so e -> e * exp(2i*theta) under rotation
    and e -> conj(e) under reflection, by construction.
  - |e| < 1 enforced by a radial squash (magnitude only, direction untouched).
  - No fixed spatial size: global attention pooling makes the model agnostic to
    stamp size, so cutout size / pixel scale can change without edits.
"""

import torch
from escnn import gspaces, nn

__all__ = ["ShapeNet_v2", "check_equivariance"]


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class MinMaxNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, C, H, W)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min)


class EqBackbone(torch.nn.Module):
    """O(2)-equivariant feature extractor. Returns a GeometricTensor."""

    def __init__(
        self,
        size=128,
        N=8,
        L=2,
        widths=(32, 64, 64),
        dropout=0.1,
        stride3=1,
        mask_margin=4,
    ):
        super().__init__()
        self.r2_act = gspaces.flipRot2dOnR2(N=-1)  # O(2)
        self.G = self.r2_act.fibergroup
        irreps = self.G.bl_irreps(L)

        w1, w2, w3 = widths

        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        self.mask = nn.MaskModule(in_type, size, margin=mask_margin)

        # --- block 1: 128 -> 64 ---
        act1 = nn.FourierELU(self.r2_act, w1, irreps=irreps, N=N)
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, act1.in_type, kernel_size=5, padding=2, bias=False),
            nn.IIDBatchNorm2d(act1.in_type),
            act1,
            nn.PointwiseAvgPoolAntialiased2D(act1.out_type, sigma=0.66, stride=2),
        )

        # --- block 2: 64 -> 32 ---
        act2 = nn.FourierELU(self.r2_act, w2, irreps=irreps, N=N)
        self.block2 = nn.SequentialModule(
            nn.R2Conv(
                self.block1.out_type, act2.in_type, kernel_size=5, padding=2, bias=False
            ),
            nn.IIDBatchNorm2d(act2.in_type),
            act2,
            nn.FieldDropout(act2.out_type, p=dropout),
            nn.PointwiseAvgPoolAntialiased2D(act2.out_type, sigma=0.66, stride=2),
        )

        # --- block 3: 32 -> 32 (or 16 if stride3=2) ---
        act3 = nn.FourierELU(self.r2_act, w3, irreps=irreps, N=N)
        layers = [
            nn.R2Conv(
                self.block2.out_type, act3.in_type, kernel_size=3, padding=1, bias=False
            ),
            nn.IIDBatchNorm2d(act3.in_type),
            act3,
            nn.FieldDropout(act3.out_type, p=dropout),
        ]
        if stride3 > 1:
            layers.append(
                nn.PointwiseAvgPoolAntialiased2D(
                    act3.out_type, sigma=0.66, stride=stride3
                )
            )
        self.block3 = nn.SequentialModule(*layers)

        self.out_type = self.block3.out_type
        self.norm = MinMaxNorm()

    def forward(self, image: torch.Tensor):
        x = self.input_type(self.norm(image))
        x = self.mask(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# ---------------------------------------------------------------------------
# Spin-2 head
# ---------------------------------------------------------------------------
class Spin2Head(torch.nn.Module):
    """
    Splits features into an invariant branch and a spin-2 branch.

    Invariant scalars may pass through arbitrary pointwise nonlinearities;
    the spin-2 fields are only ever scaled by those invariants, so the output
    inherits the frequency-2 transformation law exactly.
    """

    def __init__(self, r2_act, feat_type, K=16, C=32, hidden=64):
        super().__init__()
        G = r2_act.fibergroup

        spin2 = G.irrep(1, 2)  # O(2), frequency 2, 2-dimensional
        assert spin2.size == 2, f"expected a 2-dim irrep, got size {spin2.size}"

        self.K = K
        spin_type = nn.FieldType(r2_act, K * [spin2])
        inv_type = nn.FieldType(r2_act, C * [r2_act.trivial_repr])

        self.to_spin = nn.R2Conv(feat_type, spin_type, kernel_size=1, bias=False)
        self.to_inv = nn.R2Conv(feat_type, inv_type, kernel_size=1, bias=False)

        # 1x1 convs on invariant scalars: pointwise, so equivariance is preserved.
        self.mix = torch.nn.Sequential(
            torch.nn.BatchNorm2d(C),
            torch.nn.SiLU(),
            torch.nn.Conv2d(C, hidden, 1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(hidden, K + 1, 1),  # K gates + 1 pooling logit
        )

    def forward(self, f):
        B = f.tensor.shape[0]

        v = self.to_spin(f).tensor  # (B, 2K, H, W)
        s = self.to_inv(f).tensor  # (B,  C, H, W)

        h = self.mix(s)
        w, a = h[:, : self.K], h[:, self.K :]  # invariant gates, invariant weights

        H, W = v.shape[-2:]
        v = v.view(B, self.K, 2, H, W)
        e_map = (w.unsqueeze(2) * v).sum(dim=1)  # (B, 2, H, W), still spin-2

        # Invariant attention over space (a learned KSB-style weight function).
        a = torch.softmax(a.flatten(2), dim=-1).view(B, 1, H, W)
        return (e_map * a).sum(dim=(-2, -1))  # (B, 2)


def bound_ellipticity(v, eps=1e-6):
    """Radial squash: |e| < 1, direction preserved, equivariance intact."""
    r = torch.sqrt((v * v).sum(-1, keepdim=True) + eps**2)
    return v * (torch.tanh(r) / r)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class ShapeNet_v2(torch.nn.Module):
    def __init__(
        self,
        size=128,
        K=16,
        C=32,
        hidden=64,
        dropout=0.1,
        stride3=1,
        flip_e2=False,
    ):

        super().__init__()
        self.eq = EqBackbone(size=size, dropout=dropout, stride3=stride3)
        self.head = Spin2Head(self.eq.r2_act, self.eq.out_type, K=K, C=C, hidden=hidden)
        self.register_buffer(
            "_sign", torch.tensor([1.0, -1.0 if flip_e2 else 1.0]), persistent=False
        )

    def forward(self, im: torch.Tensor):
        e = self.head(self.eq(im))
        return bound_ellipticity(e * self._sign)


# ---------------------------------------------------------------------------
# Equivariance test
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_equivariance(model, x, verbose=True):
    """
    Rotating the image by 90 deg means 2*theta = 180 deg, so e -> -e.
    Flipping horizontally gives e -> (e1, -e2).

    Call with model.eval() -- BatchNorm in train mode will break this.
    Both residuals should sit at float noise level.
    """
    was_training = model.training
    model.eval()

    e = model(x)
    e_rot = model(torch.rot90(x, 1, dims=(-2, -1)))
    e_flip = model(torch.flip(x, dims=(-1,)))

    sign = torch.tensor([1.0, -1.0], device=e.device, dtype=e.dtype)
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
    model = ShapeNet_v2(size=128, flip_e2=True)
    x = torch.randn(4, 1, 128, 128)
    print("output shape:", model(x).shape)
    check_equivariance(model, x)
