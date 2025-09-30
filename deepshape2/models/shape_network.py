# %%
import random

import numpy as np
import torch
from escnn import gspaces, nn

# %% Define seeds
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)

# %%


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


# Full Model
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


# %%
autoencoder = torch.jit.load("/scratch/tripathi/Model_Weights/autoencoder_jit.pt")


# Full Model
class shapenet_full(torch.nn.Module):
    def __init__(self, eq_model=eq_block(), encoder=autoencoder):
        super().__init__()

        self.eq = eq_model
        self.encode = autoencoder.encoder

        c1 = 32 * 12 * 12
        c2 = 1152

        # Fully Connected classifier
        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c1 + c2),
            torch.nn.ReLU(),
            torch.nn.Linear(c1 + c2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.Tanh(),
        )

    def forward(self, input: torch.Tensor):
        im = input[:, 0, :, :].unsqueeze(1)
        psf = input[:, 1, :, :].unsqueeze(1)

        im = self.eq(im)
        psf = self.encode(psf)

        im = torch.nn.Flatten()(im)

        features = torch.cat((im, psf), dim=1)

        out = self.fully_net(features)

        return out
