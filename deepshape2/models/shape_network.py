# %%
import torch
from escnn import gspaces, nn

from ..utils.io import load_config
from ..utils.torch_utils import set_seed

set_seed()
cfg = load_config()
MODEL_DIR = cfg["MODEL_DIR"]

__all__ = ["shapenet", "shapenet_full"]
# %% Equivariant Block for feature extraction from images


class MinMaxNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, C, H, W)
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min)


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

        self.norm = MinMaxNorm()

    def forward(self, image: torch.Tensor):
        image = self.norm(image)

        x = self.input_type(image)
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


class FullNet(torch.nn.Module):
    def __init__(self, eq_dim, psf_dim):
        super().__init__()

        self.eq_proj = torch.nn.Sequential(
            torch.nn.Linear(eq_dim, eq_dim // 32),
            torch.nn.ReLU(),
        )

        self.psf_proj = torch.nn.Sequential(
            torch.nn.Linear(psf_dim, psf_dim // 32),
            torch.nn.ReLU(),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(eq_dim // 32 + psf_dim // 32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
            torch.nn.Tanh(),
        )

    def forward(self, eq_feat, psf_feat):
        eq_h = self.eq_proj(eq_feat)
        psf_h = self.psf_proj(psf_feat)
        x = torch.cat([eq_h, psf_h], dim=1)
        return self.head(x)


class shapenet_full(torch.nn.Module):
    def __init__(
        self,
        eq_block=eq_block,
        encoder_path=MODEL_DIR / "autoencoder_jit.pt",
        eq_path=MODEL_DIR / "eq_block.pt",
    ):
        super().__init__()

        self.eq = eq_block()

        if eq_path is not None:
            self.eq.load_state_dict(torch.load(eq_path, map_location="cpu"))

        autoencoder = torch.jit.load(encoder_path, map_location="cpu")
        self.encode = autoencoder.encoder
        self.encode.eval()

        # feature dimensions
        self.eq_feat_dim = 32 * 12 * 12
        self.psf_latent_dim = 1152

        # Fully Connected classifier
        # self.fully_net = FullNet(eq_dim=self.eq_feat_dim, psf_dim=self.psf_latent_dim)

        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.eq_feat_dim + self.psf_latent_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.eq_feat_dim + self.psf_latent_dim, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.Tanh(),
        )

        self.flatten = torch.nn.Flatten()
        # self.norm = MinMaxNorm()

    def forward(self, input: torch.Tensor):
        im = input[:, 0, :, :].unsqueeze(1)
        psf = input[:, 1, :, :].unsqueeze(1)

        # im = self.norm(im)
        # psf = self.norm(psf)

        psf_coded = self.encode(psf)

        im_feat = self.eq(im)
        im_feat = self.flatten(im_feat)

        # out = self.fully_net(im_feat, psf_coded)

        features = torch.cat((im_feat, psf_coded), dim=1)
        out = self.fully_net(features)

        return out
