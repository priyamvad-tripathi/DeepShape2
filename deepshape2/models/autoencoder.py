# %%
import torch.nn as nn

from deepshape2.utils import set_seed

from .vae import MultiHeadSelfAttention2D as MHA

# Seed for reproducibility
set_seed(2024)

__all__ = ["Autoender"]


class Autoender(nn.Module):
    """
    Autoencoder model for encoding PSF data
    """

    def __init__(self, attention=False):
        super().__init__()

        self.expected_image_shape = (1, 128, 128)
        self.channels = 16
        self.latent_dim = 8 * 12 * 12
        self.attention = attention

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.channels, 3, padding=1),  # (128, 128)
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                self.channels, 2 * self.channels, 3, padding=1, stride=2
            ),  # (64, 64)
            nn.ReLU(),
            nn.Conv2d(2 * self.channels, 2 * self.channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                2 * self.channels, 4 * self.channels, 3, padding=1, stride=2
            ),  # (32, 32)
            nn.ReLU(),
            nn.Conv2d(4 * self.channels, 4 * self.channels, 3, padding=1),
            MHA(4 * self.channels) if self.attention else nn.Identity(),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * self.channels * 32 * 32, self.latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 4 * self.channels * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(4 * self.channels, 32, 32)),
            nn.ConvTranspose2d(
                4 * self.channels, 4 * self.channels, 3, padding=1
            ),  # (32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(
                4 * self.channels,
                2 * self.channels,
                3,
                padding=1,
                stride=2,
                output_padding=1,
            ),  # (64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(2 * self.channels, 2 * self.channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * self.channels,
                self.channels,
                3,
                padding=1,
                stride=2,
                output_padding=1,
            ),  # (128, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(self.channels, self.channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat
