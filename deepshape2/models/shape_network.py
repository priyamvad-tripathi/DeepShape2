# %%
import torch
from escnn import gspaces, nn

from deepshape2.utils import load_config, set_seed

set_seed()
cfg = load_config()
MODEL_DIR = cfg["MODEL_DIR"]

__all__ = ["shapenet", "shapenet_full"]
# %% Equivariant Block for feature extraction from images


class eq_block(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # the model is equivariant under all planar rotations
        self.r2_act = gspaces.flipRot2dOnR2(N=-1)
        self.G = self.r2_act.fibergroup

        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        self.mask = nn.MaskModule(in_type, 128, margin=4)

        # convolution 1
        activation1 = nn.FourierELU(self.r2_act, 32, irreps=self.G.bl_irreps(2), N=8)
        out_type = activation1.in_type
        self.block1 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=4, bias=False, padding=1),
            nn.IIDBatchNorm2d(out_type),
            activation1,
            nn.PointwiseAvgPool2D(out_type, kernel_size=4),
        )

        # convolution 2
        in_type = self.block1.out_type
        activation2 = nn.FourierELU(self.r2_act, 64, irreps=self.G.bl_irreps(2), N=8)
        out_type = activation2.in_type
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=4, bias=False),
            activation2,
            nn.FieldDropout(out_type, p=0.3),
            nn.PointwiseAvgPool2D(out_type, kernel_size=2),
        )

        # convolution 3
        in_type = self.block2.out_type
        activation3 = nn.FourierELU(self.r2_act, 64, irreps=self.G.bl_irreps(2), N=8)
        out_type = activation3.in_type
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=3, bias=False),
            activation3,
        )

        # number of output invariant channels
        c = 32

        # last 1x1 convolution layer, which maps the regular fields to c=128 invariant scalar fields
        output_invariant_type = nn.FieldType(
            self.r2_act, c * [self.r2_act.trivial_repr]
        )
        self.invariant_map = nn.R2Conv(
            out_type, output_invariant_type, kernel_size=1, bias=False
        )

    def forward(self, input: torch.Tensor):
        x = self.input_type(input)
        x = self.mask(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # extract invariant features
        x = self.invariant_map(x)

        # unwrap the output GeometricTensor
        x = x.tensor
        return x


# %% Simple Model for true images: no PSF encoding
class shapenet(torch.nn.Module):
    def __init__(self, eq_block=eq_block()):
        super().__init__()

        self.eq = eq_block

        c1 = 32

        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(c1 * 12 * 12),
            torch.nn.ReLU(),
            torch.nn.Linear(c1 * 12 * 12, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.Tanh(),
        )

    def forward(self, im: torch.Tensor):
        feat = self.eq(im)
        out = self.fully_net(feat)

        return out


# %% Full Model with PSF encoding


class PSFResidualHead(torch.nn.Module):
    def __init__(self, feat_dim, psf_dim, hidden_dim=128):
        super().__init__()

        self.psf_gate = torch.nn.Sequential(
            torch.nn.Linear(psf_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feat_dim),
            torch.nn.Sigmoid(),
        )

        self.delta_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, feat, psf_latent):
        gate = self.psf_gate(psf_latent)
        feat_mod = feat * gate
        delta_e = self.delta_head(feat_mod)
        return delta_e


class shapenet_full(torch.nn.Module):
    def __init__(
        self,
        eq_block=eq_block(),
        encoder_path=MODEL_DIR + "autoencoder_jit.pt",
        eq_path=MODEL_DIR + "eq_block.pt",
    ):
        super().__init__()

        self.eq = eq_block

        if eq_path is not None:
            self.eq.load_state_dict(torch.load(eq_path, map_location="cpu"))

        autoencoder = torch.jit.load(encoder_path, map_location="cpu")
        self.encode = autoencoder.encoder
        self.encode.eval()
        for p in self.encode.parameters():
            p.requires_grad = False

        # feature dimensions
        self.eq_feat_dim = 32 * 12 * 12
        self.psf_latent_dim = 1152

        # baseline head for shape prediction
        self.base_head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.eq_feat_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.eq_feat_dim, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.Tanh(),
        )

        # PSF-conditioned residual head
        self.psf_head = PSFResidualHead(
            feat_dim=self.eq_feat_dim, psf_dim=self.psf_latent_dim
        )

    def forward(self, input):
        im = input[:, 0:1]
        psf = input[:, 1 : 1 + 1]

        feat = self.eq(im)
        feat = torch.flatten(feat, start_dim=1)

        with torch.no_grad():
            psf_latent = self.encode(psf)

        e_base = self.base_head(feat)
        delta_e = self.psf_head(feat, psf_latent)

        e_pred = e_base + delta_e
        return e_pred, e_base, delta_e
