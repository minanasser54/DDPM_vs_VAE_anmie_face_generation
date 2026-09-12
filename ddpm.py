"""
ddpm.py - Denoising Diffusion Probabilistic Model (Ho et al. 2020).
"""

import math

import torch
import torch.nn as nn

import conf


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, timesteps: int, dim: int, dim_out: int):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        ts = torch.arange(timesteps, dtype=torch.float32)
        emb = ts[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (timesteps, dim)

        self.embedding_table = nn.Embedding.from_pretrained(emb, freeze=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )

    def forward(self, t: torch.Tensor):
        return self.mlp(self.embedding_table(t))


# UNet building blocks

class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.mha = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x)
        h_ = h_.view(b, c, h * w).swapaxes(1, 2)   # (B, HW, C)
        attn_out, _ = self.mha(h_, h_, h_)
        attn_out = attn_out.swapaxes(1, 2).view(b, c, h, w)
        return x + attn_out


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_c)

        self.norm2 = nn.GroupNorm(8, out_c)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)

        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    def __init__(self, c_in=conf.CHANNELS, c_out=conf.CHANNELS,
                 base=conf.DDPM_BASE_CHANNELS, channel_mults=(1, 2, 4, 4),
                 attn_resolutions=(16, 8), num_res_blocks=2,
                 time_dim=conf.DDPM_TIME_DIM, timesteps=conf.DDPM_TIMESTEPS,
                 image_size=conf.IMAGE_SIZE, dropout=0.1):
        super().__init__()
        self.time_embed = SinusoidalPositionEmbeddings(timesteps, base, time_dim)

        self.in_conv = nn.Conv2d(c_in, base, 3, padding=1)

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        channels = [base]
        cur_c = base
        cur_res = image_size
        for level, mult in enumerate(channel_mults):
            out_c = base * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(cur_c, out_c, time_dim, dropout))
                cur_c = out_c
                if cur_res in attn_resolutions:
                    self.down_blocks.append(SelfAttention(cur_c))
                channels.append(cur_c)
            if level != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(cur_c))
                channels.append(cur_c)
                cur_res //= 2

        # --- Bottleneck ---
        self.mid_block1 = ResBlock(cur_c, cur_c, time_dim, dropout)
        self.mid_attn = SelfAttention(cur_c)
        self.mid_block2 = ResBlock(cur_c, cur_c, time_dim, dropout)

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_c = base * mult
            for _ in range(num_res_blocks + 1):
                skip_c = channels.pop()
                self.up_blocks.append(ResBlock(cur_c + skip_c, out_c, time_dim, dropout))
                cur_c = out_c
                if cur_res in attn_resolutions:
                    self.up_blocks.append(SelfAttention(cur_c))
            if level != 0:
                self.up_blocks.append(Upsample(cur_c))
                cur_res *= 2

        self.out_norm = nn.GroupNorm(8, cur_c)
        self.out_conv = nn.Conv2d(cur_c, c_out, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        h = self.in_conv(x)
        skips = [h]

        for layer in self.down_blocks:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                skips.append(h)
            elif isinstance(layer, SelfAttention):
                h = layer(h)
                skips[-1] = h  # keep skip in sync with post-attention features
            else:  # Downsample
                h = layer(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for layer in self.up_blocks:
            if isinstance(layer, ResBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)
            elif isinstance(layer, SelfAttention):
                h = layer(h)
            else:  # Upsample
                h = layer(h)

        h = self.out_conv(torch.nn.functional.silu(self.out_norm(h)))
        return h


# Diffusion process (linear beta schedule)

class Diffusion:
    """
    Holds the fixed linear noise schedule and implements:
      - q_sample: forward diffusion, x_0 -> x_t in closed form
      - sample:   full reverse loop from pure noise x_T -> x_0 (ancestral
                  sampling, Algorithm 2 in Ho et al. 2020)

    All schedule tensors live on `device` from construction so there's no
    dtype/device juggling at train or sample time (a common source of subtle
    DDPM bugs).
    """

    def __init__(self, timesteps=conf.DDPM_TIMESTEPS, beta_start=conf.DDPM_BETA_START,
                 beta_end=conf.DDPM_BETA_END, image_size=conf.IMAGE_SIZE,
                 channels=conf.CHANNELS, device=conf.DEVICE):
        self.timesteps = timesteps
        self.image_size = image_size
        self.channels = channels
        self.device = device

        beta = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        self.beta = beta
        self.alpha = alpha
        self.alpha_hat = alpha_hat
        self.sqrt_alpha_hat = torch.sqrt(alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - alpha_hat)
        self.sqrt_recip_alpha = 1.0 / torch.sqrt(alpha)
        self.sqrt_beta = torch.sqrt(beta)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor):
        """Gather per-sample scalars from a (timesteps,) schedule tensor and
        reshape to broadcast against an (B, C, H, W) image batch."""
        return arr.gather(0, t).view(-1, 1, 1, 1)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward process closed form: x_t = sqrt(alpha_hat_t) * x0 + sqrt(1 - alpha_hat_t) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        mean = self._extract(self.sqrt_alpha_hat, t) * x0
        std = self._extract(self.sqrt_one_minus_alpha_hat, t)
        return mean + std * noise, noise

    def sample_timesteps(self, batch_size: int):
        return torch.randint(low=0, high=self.timesteps, size=(batch_size,), device=self.device)

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, channels=None, image_size=None, verbose=True):
        """Full reverse diffusion loop (ancestral sampling). Returns images
        in [-1, 1] range (same as training data)."""
        channels = channels or self.channels
        image_size = image_size or self.image_size

        was_training = model.training
        model.eval()
        x = torch.randn(n, channels, image_size, image_size, device=self.device)

        iterator = reversed(range(self.timesteps))
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=self.timesteps, desc="sampling")

        for i in iterator:
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t)

            sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alpha, t)
            beta_t = self._extract(self.beta, t)
            sqrt_one_minus_alpha_hat_t = self._extract(self.sqrt_one_minus_alpha_hat, t)
            sqrt_beta_t = self._extract(self.sqrt_beta, t)

            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

            x = sqrt_recip_alpha_t * (
                x - (beta_t / sqrt_one_minus_alpha_hat_t) * predicted_noise
            ) + sqrt_beta_t * noise

        if was_training:
            model.train()
        return x.clamp(-1, 1)


if __name__ == "__main__":
    # quick shape + gradient sanity check
    model = UNet()
    diffusion = Diffusion(device="cpu")
    x = torch.randn(2, conf.CHANNELS, conf.IMAGE_SIZE, conf.IMAGE_SIZE)
    t = diffusion.sample_timesteps(2)
    x_t, noise = diffusion.q_sample(x, t)
    pred = model(x_t, t)
    loss = torch.nn.functional.mse_loss(pred, noise)
    loss.backward()
    print("x_t:", x_t.shape, "pred noise:", pred.shape, "loss:", loss.item())
    print("params:", sum(p.numel() for p in model.parameters()))
    # confirm every parameter got a gradient (catches dead/unused branches)
    n_no_grad = sum(1 for p in model.parameters() if p.grad is None)
    print("params with no grad:", n_no_grad)
