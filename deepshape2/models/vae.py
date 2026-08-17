# %%Import Libraries

import torch
import torch.nn as nn

from ..utils.torch_utils import set_seed

# Seed for reproducibility
set_seed(2024)

__all__ = ["VAE", "MultiHeadSelfAttention2D", "VAE_mask"]


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
        activation="soft",
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
class VAE_mask(nn.Module):
    """
    Same encoder/decoder topology as VAE, but the decoder head predicts a
    per-pixel multiplicative mask in [0, m_max] which is applied to the input
    in linear flux.

    Args:
        latent_dim: latent width (16 matches the working VAE).
        bias, attention, variational: as in VAE.
        m_max: upper bound on the mask.  1.0 enforces recon <= input, which is
            physically correct for additive blends.  Values slightly above 1
            (e.g. 1.05) give flux headroom if the deblender is allowed to
            over-recover; only relax this if you see a systematic T_ratio < 1.
        clamp_asinh: input is clamped to +/- this before sinh() to avoid
            overflow.  sinh(12) ~ 8e4, comfortably above any real stamp value.
        identity_init: zero the final conv weights and set bias so the initial
            mask is uniformly sigmoid(init_bias) ~ 0.98.  This starts training
            at (near) identity, so the network only has to learn where to
            subtract.  Gradients still flow, since the layer input is nonzero.
        init_bias: pre-sigmoid bias for identity_init.  4.0 -> 0.982.
    """

    # flag consumed by _decode_from_mu so it never falls through to
    # self.decoder(mu) and silently return a mask as if it were an image
    mask_mode = True

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

        # Mask must be bounded in [0, 1]; sigmoid is not optional here.
        self.activation = nn.Sigmoid()

        # ---------------- Encoder ----------------
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

        if self.variational:
            self.mu = nn.Linear(self.latent_dim_1, self.latent_dim)
            self.logvar = nn.Linear(self.latent_dim_1, self.latent_dim)
        else:
            self.latent = nn.Linear(self.latent_dim_1, self.latent_dim)

        # ---------------- Decoder ----------------
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim_1),
            nn.BatchNorm1d(self.latent_dim_1),
            nn.PReLU(),
            nn.Linear(self.latent_dim_1, 32 * self.channels * 4 * 4),
            nn.PReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32 * self.channels, 4, 4)),
            # (4, 4) -> (8, 8)
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
            # (8, 8) -> (16, 16)
            nn.ConvTranspose2d(
                16 * self.channels,
                8 * self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.PReLU(),
            # (16, 16) -> (32, 32)
            nn.ConvTranspose2d(
                8 * self.channels,
                4 * self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.PReLU(),
            # (32, 32) -> (64, 64)
            nn.ConvTranspose2d(
                4 * self.channels,
                2 * self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.PReLU(),
            # (64, 64) -> (128, 128)
            nn.ConvTranspose2d(
                2 * self.channels,
                self.channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=self.bias,
            ),
            nn.PReLU(),
            # Final: (128, 128) mask logits -> sigmoid
            nn.ConvTranspose2d(
                self.channels, 1, kernel_size=3, padding=1, bias=self.bias
            ),
            self.activation,
        )

        if identity_init:
            self._identity_init(init_bias)

    # ------------------------------------------------------------------
    def _identity_init(self, init_bias=4.0):
        """
        Start at (near) identity: zero the final conv weights so the mask is
        uniform regardless of input, and bias it to sigmoid(init_bias) ~ 0.98.

        Zeroing the weights matters as much as the bias -- otherwise the
        starting mask is identity plus noise, and the noise is exactly the
        shape error you are trying to eliminate on isolated stamps.
        """
        final = self.decoder[-2]  # ConvTranspose2d before the Sigmoid
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.constant_(final.bias, float(init_bias))

    def _apply_mask(self, x, m):
        """recon_asinh = asinh(m_max * m * sinh(x_asinh))."""
        xc = x.clamp(-self.clamp_asinh, self.clamp_asinh)
        return torch.asinh(self.m_max * m * torch.sinh(xc))

    # ------------------------------------------------------------------
    def reparameterize(self, z):
        mu = self.mu(z)
        logvar = self.logvar(z)
        std = torch.exp(0.5 * logvar) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std, mu, logvar

    def encode(self, x):
        """Returns (z, mu, logvar); mu/logvar are None when non-variational."""
        z1 = self.encoder(x)
        if self.variational:
            return self.reparameterize(z1)
        return self.latent(z1), None, None

    def predict_mask(self, x):
        """The mask itself, for diagnostics. Deterministic (uses mu)."""
        z1 = self.encoder(x)
        z = self.mu(z1) if self.variational else self.latent(z1)
        return self.decoder(z)

    def decode(self, z, x=None):
        """
        Decode a latent into a reconstruction.  The input image is REQUIRED:
        the model predicts a mask, and the mask is meaningless without the
        image it multiplies.
        """
        m = self.decoder(z)
        if x is None:
            raise ValueError(
                "VAE_mask.decode needs the input image (recon = mask * input)."
            )
        return self._apply_mask(x, m)

    def forward(self, x):
        z1 = self.encoder(x)
        if self.variational:
            z, mu, logvar = self.reparameterize(z1)
        else:
            z = self.latent(z1)

        m = self.decoder(z)
        xhat = self._apply_mask(x, m)

        if self.variational:
            return xhat, mu, logvar
        return xhat


# ======================================================================
# Patched deterministic-decode helper.  Replace the version in the
# evaluation module, and pass `inp` at both call sites:
#     _decode_from_mu(model, mu, inp)
# ======================================================================
def _decode_from_mu(model, mu, inp=None):
    """
    Deterministic decode (z = mu).  Returns None if unavailable.

    The mask_mode guard is essential: for VAE_mask, model.decoder(mu) returns
    the MASK, not an image.  Without the guard a silent fallback would produce
    plausible-looking arrays and nonsense PSNR.
    """
    fn = getattr(model, "decode", None)
    if fn is not None:
        if inp is not None:
            try:
                return fn(mu, inp)
            except TypeError:
                pass
            except Exception:
                return None
        try:
            return fn(mu)
        except Exception:
            pass

    if getattr(model, "mask_mode", False):
        return None  # never fall through in mask mode

    dec = getattr(model, "decoder", None)
    if dec is not None:
        try:
            return dec(mu)
        except Exception:
            pass
    return None


# ======================================================================
# Optional: mask diagnostics for the per-epoch log.
# ======================================================================
@torch.no_grad()
def mask_stats(model, inp, target, clamp=12.0):
    """
    Recover the mask from a forward pass and summarise it.

    On isolated stamps the mask should be ~1 almost everywhere; frac>0.99
    climbing toward the isolated fraction (~0.78) is the identity behaviour
    this formulation is meant to produce.
    """
    m = model.predict_mask(inp)
    on = target > 0
    return {
        "m_on": float(m[on].mean()) if on.any() else float("nan"),
        "m_off": float(m[~on].mean()) if (~on).any() else float("nan"),
        "frac_gt99": float((m > 0.99).float().mean()),
        "m_min": float(m.min()),
    }
