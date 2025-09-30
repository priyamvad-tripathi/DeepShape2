# %%Import Libraries
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Seed for reproducibility
torch.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# %% VAE Model Definition
class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # learnable scale

    def forward(self, x):
        B, C, H, W = x.shape
        query = self.query(x).view(B, -1, H * W)  # B, C//8, N
        key = self.key(x).view(B, -1, H * W)  # B, C//8, N
        value = self.value(x).view(B, -1, H * W)  # B, C, N

        attention = torch.bmm(query.permute(0, 2, 1), key)  # B, N, N
        attention = F.softmax(attention / (key.shape[1] ** 0.5), dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # B, C, N
        out = out.view(B, C, H, W)

        return self.gamma * out + x


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
        encoder_only=False,
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
        else:
            self.activation = nn.Sigmoid()
        self.bias = bias
        self.attention = attention
        self.encoder_only = encoder_only

        # Encoder
        self.encoder = nn.Sequential(
            # (128, 128)
            nn.Conv2d(1, self.channels, 3, padding=1, bias=self.bias),
            # SelfAttention2D(self.channels) if self.attention else nn.Identity(),
            nn.PReLU(),
            # (64, 64)
            nn.Conv2d(
                self.channels, 2 * self.channels, 3, padding=1, stride=2, bias=self.bias
            ),
            # SelfAttention2D(2 * self.channels) if self.attention else nn.Identity(),
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
            # SelfAttention2D(4 * self.channels) if self.attention else nn.Identity(),
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
            # SelfAttention2D(8 * self.channels) if self.attention else nn.Identity(),
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
            # SelfAttention2D(32 * self.channels) if self.attention else nn.Identity(),
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

        # Reparameterization trick
        self.mu = nn.Linear(self.latent_dim_1, self.latent_dim)
        self.logvar = nn.Linear(self.latent_dim_1, self.latent_dim)

        # Decoder
        if not self.encoder_only:
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
                # SelfAttention2D(8 * self.channels) if self.attention else nn.Identity(),
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
                # SelfAttention2D(4 * self.channels) if self.attention else nn.Identity(),
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
                # SelfAttention2D(2 * self.channels) if self.attention else nn.Identity(),
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
                # SelfAttention2D(self.channels) if self.attention else nn.Identity(),
                nn.PReLU(),
                # Final output: (128, 128)
                nn.ConvTranspose2d(
                    self.channels, 1, kernel_size=3, padding=1, bias=self.bias
                ),
                self.activation,
            )
        else:
            self.decoder = None

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
        z, mu, logvar = self.reparameterize(z1)

        if self.encoder_only:
            return z, mu, logvar

        xhat = self.decoder(z)
        return xhat, mu, logvar
        return xhat, mu, logvar
        return xhat, mu, logvar
        return xhat, mu, logvar
