import torch
import torch.nn as nn

__all__ = ["RefineNet"]


class CondMLP(nn.Module):
    def __init__(self, in_channels, n_noise_scale=10, hidden=64):
        super().__init__()
        # we take noise index -> learnable embedding -> MLP
        self.embed = nn.Embedding(n_noise_scale, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels * 3),  # gamma, beta, alpha
        )

    def forward(self, x, noise_scale_idx):
        # x: (B, C, H, W), noise_scale_idx: (B,)
        bsz, C = x.shape[0], x.shape[1]
        e = self.embed(noise_scale_idx)  # (B, hidden)
        out = self.mlp(e)  # (B, C*3)
        out = out.view(bsz, 3, C, 1, 1)  # (B, 3, C, 1, 1)
        gamma, beta, alpha = out[:, 0], out[:, 1], out[:, 2]  # each (B,C,1,1)
        return gamma, beta, alpha


class CondInstanceNorm(nn.Module):
    def __init__(self, in_channels, n_noise_scale=10, eps=0):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_noise_scale, in_channels))
        self.beta = nn.Parameter(torch.zeros(n_noise_scale, in_channels))
        self.alpha = nn.Parameter(torch.zeros(n_noise_scale, in_channels))
        self.eps = eps

    def forward(self, x, noise_scale_idx):
        # x: (batch_size, in_channels, height, width)
        # noise_scale_idx: (batch_size)
        # gamma: (n_noise_scale, in_channels)

        # bsz = x.shape[0]
        # gamma = self.gamma[noise_scale_idx].view(
        #     bsz, -1, 1, 1
        # )  # (bsz, in_channels, 1, 1)
        # beta = self.beta[noise_scale_idx].view(bsz, -1, 1, 1)
        # alpha = self.alpha[noise_scale_idx].view(bsz, -1, 1, 1)
        gamma, beta, alpha = CondMLP(x, noise_scale_idx)

        mu = x.mean(dim=(2, 3), keepdim=True)  # (batch_size, in_channels, 1, 1)
        var = x.var(dim=(2, 3), keepdim=True)  # (batch_size, in_channels, 1, 1)
        sigma = torch.sqrt(var + self.eps)  # (batch_size, in_channels, 1, 1)

        x = (x - mu) / sigma  # (batch_size, in_channels, height, width)
        x = gamma * x + beta  # (batch_size, in_channels, height, width)

        m = mu.mean(dim=1, keepdim=True)  # (batch_size, 1, 1, 1)
        if mu.shape[1] == 1:
            s = torch.ones_like(mu)
        else:
            v = mu.var(dim=1, keepdim=True)  # (batch_size, 1, 1, 1)
            s = torch.sqrt(v + self.eps)  # (batch_size, 1, 1, 1)

        x = x + alpha * (mu - m) / s  # (batch_size, in_channels, height, width)

        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, channels, norm=True, kernel_size=3, n_noise_scale=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding="same"
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding="same"
        )
        self.norm1 = CondInstanceNorm(channels, n_noise_scale) if norm else None
        self.norm2 = CondInstanceNorm(channels, n_noise_scale) if norm else None
        self.act = nn.ELU()

    def forward(self, x, noise_scale_idx):
        # x: (batch_size, in_channels, height, width)

        h = self.norm1(x, noise_scale_idx) if self.norm1 is not None else x
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h, noise_scale_idx) if self.norm2 is not None else h
        h = self.act(h)
        h = self.conv2(h)

        return x + h


class StridedConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_noise_scale=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=2,
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        self.norm1 = CondInstanceNorm(in_channels, n_noise_scale)
        self.norm2 = CondInstanceNorm(out_channels, n_noise_scale)
        self.act = nn.ELU()

    def forward(self, x, noise_scale_idx):
        # x: (batch_size, in_channels, height, width)

        h = self.norm1(x, noise_scale_idx)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h, noise_scale_idx)
        h = self.act(h)
        h = self.conv2(h)

        return h


class DilatedConvUnit(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, dilation=2, n_noise_scale=10
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation,
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        self.norm1 = CondInstanceNorm(in_channels, n_noise_scale)
        self.norm2 = CondInstanceNorm(out_channels, n_noise_scale)
        self.act = nn.ELU()

    def forward(self, x, noise_scale_idx):
        # x: (batch_size, in_channels, height, width)

        h = self.norm1(x, noise_scale_idx)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h, noise_scale_idx)
        h = self.act(h)
        h = self.conv2(h)

        return h


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers=2,
        downsample="stride",
        dilation=2,
        n_noise_scale=10,
    ):
        assert downsample in ["stride", "dilation"]
        super().__init__()
        self.downsample = downsample
        self.main = nn.ModuleList([])
        for _ in range(n_layers):
            self.main.append(ResidualConvUnit(in_channels, n_noise_scale=n_noise_scale))

        if downsample == "stride":
            self.main.append(
                StridedConvUnit(in_channels, out_channels, n_noise_scale=n_noise_scale)
            )
        elif downsample == "dilation":
            self.main.append(
                DilatedConvUnit(
                    in_channels,
                    out_channels,
                    dilation=dilation,
                    n_noise_scale=n_noise_scale,
                )
            )

    def forward(self, x, noise_scale_idx):
        # x: (batch_size, in_channels, height, width)

        for layer in self.main:
            x = layer(x, noise_scale_idx)

        return x


class AdaptiveConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_noise_scale=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = ResidualConvUnit(out_channels, n_noise_scale=n_noise_scale)
        self.conv3 = ResidualConvUnit(out_channels, n_noise_scale=n_noise_scale)

    def forward(self, x, noise_scale_idx):
        # x: (batch_size, in_channels, height, width)

        h = self.conv1(x)
        h = self.conv2(h, noise_scale_idx)
        h = self.conv3(h, noise_scale_idx)

        return h


class MultiResolutionFusion(nn.Module):
    def __init__(self, channels, n_noise_scale=10):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding="same")
        self.norm1 = CondInstanceNorm(channels, n_noise_scale)
        self.norm2 = CondInstanceNorm(channels, n_noise_scale)

    def forward(self, x, y=None, noise_scale_idx=0):
        if y is None:
            return x
        else:
            h1 = self.norm1(x, noise_scale_idx)
            h1 = self.conv1(h1)

            h2 = self.norm2(y, noise_scale_idx)
            h2 = self.conv2(h2)

            return h1 + h2


class ResidualPoolingBlock(nn.Module):
    def __init__(self, channels, n_noise_scale=10):
        super().__init__()
        self.norm1 = CondInstanceNorm(channels, n_noise_scale)
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.norm2 = CondInstanceNorm(channels, n_noise_scale)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding="same")

    def forward(self, x, noise_scale_idx):
        h = self.norm1(x, noise_scale_idx)
        h = self.pool1(h)
        h = self.norm2(h, noise_scale_idx)
        h = self.conv1(h)

        return h


class ChainedResidualPool(nn.Module):
    def __init__(self, channels, n_noise_scale=10):
        super().__init__()
        self.act = nn.ELU()
        self.pool1 = ResidualPoolingBlock(channels, n_noise_scale=n_noise_scale)
        self.pool2 = ResidualPoolingBlock(channels, n_noise_scale=n_noise_scale)

    def forward(self, x, noise_scale_idx):
        x = self.act(x)
        h = self.pool1(x, noise_scale_idx)
        x = x + h
        h = self.pool2(h, noise_scale_idx)
        x = x + h

        return x


class RefineNetBlock(nn.Module):
    def __init__(self, x1_in, x2_in, channels, n_noise_scale=10):
        super().__init__()
        self.adap_x1 = AdaptiveConvBlock(x1_in, channels, n_noise_scale=n_noise_scale)
        self.adap_x2 = AdaptiveConvBlock(x2_in, channels, n_noise_scale=n_noise_scale)

        self.fusion = MultiResolutionFusion(channels, n_noise_scale=n_noise_scale)
        self.pool = ChainedResidualPool(channels, n_noise_scale=n_noise_scale)

        self.out = ResidualConvUnit(channels, n_noise_scale=n_noise_scale)

    def forward(self, x1, x2=None, noise_scale_idx=0):
        h1 = self.adap_x1(x1, noise_scale_idx)
        h2 = self.adap_x2(x2, noise_scale_idx) if x2 is not None else None
        h = self.fusion(h1, h2, noise_scale_idx)
        h = self.pool(h, noise_scale_idx)
        h = self.out(h, noise_scale_idx)

        return h


class RefineNet(nn.Module):
    def __init__(
        self, in_channels=1, hidden_channels=(128, 256, 512, 1024), n_noise_scale=10
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            in_channels,
            hidden_channels[0],
            n_layers=2,
            downsample="stride",
            n_noise_scale=n_noise_scale,
        )
        self.res2 = ResidualBlock(
            hidden_channels[0],
            hidden_channels[1],
            n_layers=2,
            downsample="dilation",
            dilation=2,
            n_noise_scale=n_noise_scale,
        )
        self.res3 = ResidualBlock(
            hidden_channels[1],
            hidden_channels[2],
            n_layers=2,
            downsample="dilation",
            dilation=4,
            n_noise_scale=n_noise_scale,
        )
        self.res4 = ResidualBlock(
            hidden_channels[2],
            hidden_channels[3],
            n_layers=2,
            downsample="dilation",
            dilation=8,
            n_noise_scale=n_noise_scale,
        )

        self.refine1 = RefineNetBlock(
            x1_in=hidden_channels[-1],
            x2_in=hidden_channels[-1],
            channels=hidden_channels[-1],
            n_noise_scale=n_noise_scale,
        )
        self.refine2 = RefineNetBlock(
            x1_in=hidden_channels[-2],
            x2_in=hidden_channels[-1],
            channels=hidden_channels[-2],
            n_noise_scale=n_noise_scale,
        )
        self.refine3 = RefineNetBlock(
            x1_in=hidden_channels[-3],
            x2_in=hidden_channels[-2],
            channels=hidden_channels[-3],
            n_noise_scale=n_noise_scale,
        )
        self.refine4 = RefineNetBlock(
            x1_in=hidden_channels[-4],
            x2_in=hidden_channels[-3],
            channels=hidden_channels[-4],
            n_noise_scale=n_noise_scale,
        )

        self.up_norm = CondInstanceNorm(hidden_channels[-4], n_noise_scale)
        self.up_conv = nn.ConvTranspose2d(
            hidden_channels[-4],
            hidden_channels[-4],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.out = AdaptiveConvBlock(
            hidden_channels[-4], in_channels, n_noise_scale=n_noise_scale
        )

    def forward(self, x, noise_scale_idx):
        h1 = self.res1(x, noise_scale_idx)
        h2 = self.res2(h1, noise_scale_idx)
        h3 = self.res3(h2, noise_scale_idx)
        h4 = self.res4(h3, noise_scale_idx)

        h = self.refine1(h4, x2=None, noise_scale_idx=noise_scale_idx)
        h = self.refine2(h3, h, noise_scale_idx)
        h = self.refine3(h2, h, noise_scale_idx)
        h = self.refine4(h1, h, noise_scale_idx)

        h = self.up_norm(h, noise_scale_idx)
        h = self.up_conv(h)
        h = self.out(h, noise_scale_idx)

        return h
