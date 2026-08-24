# %%Import Libraries

import torch
import torch.nn as nn

from ..utils.torch_utils import set_seed

# Seed for reproducibility
set_seed(2024)

__all__ = ["VAE", "MultiHeadSelfAttention2D", "VAE_skip"]


# %% VAE Model Definition
class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        assert in_channels % num_heads == 0, (
            "in_channels must be divisible by num_heads"
        )

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim**-0.5

        # Shared projection layers for all heads (Q, K, V combined)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)

        # Output projection after concatenation
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable residual scale

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, 3C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, N)  # [B, 3, H, D, N]
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: [B, num_heads, head_dim, N]

        # Transpose for attention computation: [B, num_heads, N, head_dim]
        q = q.permute(0, 1, 3, 2)  # [B, num_heads, N, head_dim]
        k = k.permute(0, 1, 3, 2)  # [B, num_heads, N, head_dim]
        v = v.permute(0, 1, 2, 3)  # [B, num_heads, head_dim, N]

        # Attention scores: [B, num_heads, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values: [B, num_heads, head_dim, N]
        out = torch.matmul(v, attn.transpose(-2, -1))  # [B, num_heads, head_dim, N]

        # Reshape back to image: [B, C, H, W]
        out = out.reshape(B, C, H, W)

        out = self.out_proj(out)  # Final projection
        return self.gamma * out + x  # Residual connection


class VAE(nn.Module):
    """
    VAE model for galaxy deblending.
    """

    def __init__(
        self,
        latent_dim=16,
        activation="relu",
        bias=True,
        attention=True,
        variational=True,
    ):
        super().__init__()

        self.expected_image_shape = (1, 128, 128)
        self.channels = 16
        self.latent_dim_1 = 512
        self.latent_dim = latent_dim
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "identity":
            self.activation = nn.Identity()
        elif activation == "soft":
            self.activation = nn.Softplus(beta=10.0)
        else:
            self.activation = nn.Sigmoid()
        self.bias = bias
        self.attention = attention
        self.variational = variational

        # Encoder
        self.encoder = nn.Sequential(
            # (128, 128)
            nn.Conv2d(1, self.channels, 3, padding=1, bias=self.bias),
            nn.PReLU(),
            # (64, 64)
            nn.Conv2d(
                self.channels, 2 * self.channels, 3, padding=1, stride=2, bias=self.bias
            ),
            nn.PReLU(),
            nn.Dropout(0.3),
            # (32, 32)
            nn.Conv2d(
                2 * self.channels,
                4 * self.channels,
                3,
                padding=1,
                stride=2,
                bias=self.bias,
            ),
            nn.PReLU(),
            # (16, 16)
            nn.Conv2d(
                4 * self.channels,
                8 * self.channels,
                3,
                padding=1,
                stride=2,
                bias=self.bias,
            ),
            nn.PReLU(),
            nn.Dropout(0.3),
            # (8, 8)
            nn.Conv2d(
                8 * self.channels,
                16 * self.channels,
                3,
                padding=1,
                stride=2,
                bias=self.bias,
            ),
            MultiHeadSelfAttention2D(16 * self.channels)
            if self.attention
            else nn.Identity(),
            nn.PReLU(),
            # (4, 4)
            nn.Conv2d(
                16 * self.channels,
                32 * self.channels,
                3,
                padding=1,
                stride=2,
                bias=self.bias,
            ),
            MultiHeadSelfAttention2D(32 * self.channels)
            if self.attention
            else nn.Identity(),
            nn.PReLU(),
            # Dense Layers
            nn.Flatten(),
            nn.Linear(32 * self.channels * 4 * 4, self.latent_dim_1),
            nn.BatchNorm1d(self.latent_dim_1),
            nn.Dropout(0.3),
            nn.PReLU(),
        )

        # Only used if variational=True
        if self.variational:
            self.mu = nn.Linear(self.latent_dim_1, self.latent_dim)
            self.logvar = nn.Linear(self.latent_dim_1, self.latent_dim)
        else:
            self.latent = nn.Linear(self.latent_dim_1, self.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim_1),
            nn.BatchNorm1d(self.latent_dim_1),
            nn.PReLU(),
            nn.Linear(self.latent_dim_1, 32 * self.channels * 4 * 4),
            nn.PReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32 * self.channels, 4, 4)),
            # (4, 4) → (8, 8)
            nn.ConvTranspose2d(
                32 * self.channels,
                16 * self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            MultiHeadSelfAttention2D(16 * self.channels)
            if self.attention
            else nn.Identity(),
            nn.PReLU(),
            # (8, 8) → (16, 16)
            nn.ConvTranspose2d(
                16 * self.channels,
                8 * self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            # MultiHeadSelfAttention2D(8 * self.channels)
            # if self.attention
            # else nn.Identity(),
            nn.PReLU(),
            # (16, 16) → (32, 32)
            nn.ConvTranspose2d(
                8 * self.channels,
                4 * self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.PReLU(),
            # (32, 32) → (64, 64)
            nn.ConvTranspose2d(
                4 * self.channels,
                2 * self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.PReLU(),
            # (64, 64) → (128, 128)
            nn.ConvTranspose2d(
                2 * self.channels,
                self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.PReLU(),
            # Final output: (128, 128)
            nn.ConvTranspose2d(
                self.channels, 1, kernel_size=3, padding=1, bias=self.bias
            ),
            self.activation,
        )

    def reparameterize(self, z):
        mu = self.mu(z)
        logvar = self.logvar(z)

        std = (
            torch.exp(0.5 * logvar) + 1e-6
        )  # Added small constant for numerical stability
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar

    def forward(self, x):
        z1 = self.encoder(x)

        if self.variational:
            z, mu, logvar = self.reparameterize(z1)
        else:
            z = self.latent(z1)

        xhat = self.decoder(z)

        if self.variational:
            return xhat, mu, logvar
        else:
            return xhat


# %%
class VAE_skip(nn.Module):
    """
    Deblender predicting a per-pixel multiplicative mask, applied to the input
    in linear flux:

        recon_asinh = asinh(m_max * m * sinh(x_asinh))

    Blends are additive, so target <= input pointwise and the optimal mask
    m* = target_lin / input_lin lies in [0, 1].  The identity map (m = 1) is
    therefore trivially reachable, which matters because ~78% of stamps are
    near-isolated.  Background pixels come out exactly zero for free.

    The mask head sees the decoder's full-resolution features concatenated
    with the raw input image; the decoder alone reaches 128x128 only after
    upsampling from 4x4, which limits the detail it can put in the mask.

    Args:
        latent_dim: latent width.
        m_max: mask upper bound.  1.0 enforces recon <= input.
        clamp_asinh: input clamp before sinh(), guards overflow.
        identity_init: start at m ~ sigmoid(init_bias) everywhere.
        input_skip: False reverts to the no-skip head (loads older
            VAE_mask checkpoints).
    """

    mask_mode = True  # read by _decode_from_mu

    def __init__(
        self,
        latent_dim=16,
        bias=True,
        attention=True,
        variational=True,
        m_max=1.0,
        clamp_asinh=12.0,
        identity_init=True,
        init_bias=4.0,
        input_skip=True,
        skip_hidden=32,
    ):
        super().__init__()

        self.expected_image_shape = (1, 128, 128)
        self.channels = 16
        self.latent_dim_1 = 512
        self.latent_dim = latent_dim
        self.bias = bias
        self.attention = attention
        self.variational = variational
        self.m_max = float(m_max)
        self.clamp_asinh = float(clamp_asinh)
        self.input_skip = bool(input_skip)

        C = self.channels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1, bias=bias),
            nn.PReLU(),
            nn.Conv2d(C, 2 * C, 3, padding=1, stride=2, bias=bias),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(2 * C, 4 * C, 3, padding=1, stride=2, bias=bias),
            nn.PReLU(),
            nn.Conv2d(4 * C, 8 * C, 3, padding=1, stride=2, bias=bias),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(8 * C, 16 * C, 3, padding=1, stride=2, bias=bias),
            MultiHeadSelfAttention2D(16 * C) if attention else nn.Identity(),
            nn.PReLU(),
            nn.Conv2d(16 * C, 32 * C, 3, padding=1, stride=2, bias=bias),
            MultiHeadSelfAttention2D(32 * C) if attention else nn.Identity(),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(32 * C * 4 * 4, self.latent_dim_1),
            nn.BatchNorm1d(self.latent_dim_1),
            nn.Dropout(0.3),
            nn.PReLU(),
        )

        if variational:
            self.mu = nn.Linear(self.latent_dim_1, latent_dim)
            self.logvar = nn.Linear(self.latent_dim_1, latent_dim)
        else:
            self.latent = nn.Linear(self.latent_dim_1, latent_dim)

        dec = [
            nn.Linear(latent_dim, self.latent_dim_1),
            nn.BatchNorm1d(self.latent_dim_1),
            nn.PReLU(),
            nn.Linear(self.latent_dim_1, 32 * C * 4 * 4),
            nn.PReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32 * C, 4, 4)),
            nn.ConvTranspose2d(32 * C, 16 * C, 4, stride=2, padding=1, bias=bias),
            MultiHeadSelfAttention2D(16 * C) if attention else nn.Identity(),
            nn.PReLU(),
            nn.ConvTranspose2d(16 * C, 8 * C, 4, stride=2, padding=1, bias=bias),
            nn.PReLU(),
            nn.ConvTranspose2d(8 * C, 4 * C, 4, stride=2, padding=1, bias=bias),
            nn.PReLU(),
            nn.ConvTranspose2d(4 * C, 2 * C, 4, stride=2, padding=1, bias=bias),
            nn.PReLU(),
            nn.ConvTranspose2d(2 * C, C, 4, stride=2, padding=1, bias=bias),
            nn.PReLU(),
        ]

        if self.input_skip:
            self.mask_head = nn.Sequential(
                nn.Conv2d(C + 1, skip_hidden, 3, padding=1, bias=bias),
                nn.PReLU(),
                nn.Conv2d(skip_hidden, skip_hidden, 3, padding=1, bias=bias),
                nn.PReLU(),
                nn.Conv2d(skip_hidden, 1, 3, padding=1, bias=bias),
            )
        else:
            dec += [
                nn.ConvTranspose2d(C, 1, kernel_size=3, padding=1, bias=bias),
                nn.Sigmoid(),
            ]
            self.mask_head = None

        self.decoder = nn.Sequential(*dec)

        if identity_init:
            self._identity_init(init_bias)

    def _identity_init(self, init_bias=4.0):
        """Zero the final weights so the initial mask is uniform, not noisy."""
        final = self.mask_head[-1] if self.input_skip else self.decoder[-2]
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.constant_(final.bias, float(init_bias))

    def _mask(self, z, x):
        h = self.decoder(z)
        if not self.input_skip:
            return h
        return torch.sigmoid(self.mask_head(torch.cat([h, x], dim=1)))

    def _apply_mask(self, x, m):
        xc = x.clamp(-self.clamp_asinh, self.clamp_asinh)
        return torch.asinh(self.m_max * m * torch.sinh(xc))

    def reparameterize(self, z):
        mu = self.mu(z)
        logvar = self.logvar(z)
        std = torch.exp(0.5 * logvar) + 1e-6
        return mu + torch.randn_like(std) * std, mu, logvar

    def predict_mask(self, x):
        """The mask itself, for diagnostics. Deterministic (uses mu)."""
        z1 = self.encoder(x)
        z = self.mu(z1) if self.variational else self.latent(z1)
        return self._mask(z, x)

    def decode(self, z, x=None):
        if x is None:
            raise ValueError("VAE_skip.decode needs the input image.")
        return self._apply_mask(x, self._mask(z, x))

    def forward(self, x):
        z1 = self.encoder(x)
        if self.variational:
            z, mu, logvar = self.reparameterize(z1)
        else:
            z = self.latent(z1)

        xhat = self._apply_mask(x, self._mask(z, x))
        return (xhat, mu, logvar) if self.variational else xhat
