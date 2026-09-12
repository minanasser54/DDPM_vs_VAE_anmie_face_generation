import torch
import torch.nn as nn
import torch.nn.functional as F

import conf


class Encoder(nn.Module):
    """4x stride-2 convs: 64 -> 32 -> 16 -> 8 -> 4, then flatten to mu/logvar."""

    def __init__(self, in_channels=conf.CHANNELS, base=conf.VAE_BASE_CHANNELS, latent_dim=conf.VAE_LATENT_DIM):
        super().__init__()
        self.base = base
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, stride=2, padding=1),       # 64 -> 32
            nn.BatchNorm2d(base), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),          # 32 -> 16
            nn.BatchNorm2d(base * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),      # 16 -> 8
            nn.BatchNorm2d(base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1),      # 8 -> 4
            nn.BatchNorm2d(base * 8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = base * 8 * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Mirror of the encoder: fc -> 4x stride-2 transposed convs, tanh output."""

    def __init__(self, out_channels=conf.CHANNELS, base=conf.VAE_BASE_CHANNELS, latent_dim=conf.VAE_LATENT_DIM):
        super().__init__()
        self.base = base
        self.fc = nn.Linear(latent_dim, base * 8 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1),  # 4 -> 8
            nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),      # 16 -> 32
            nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, out_channels, 4, stride=2, padding=1),  # 32 -> 64
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base * 8, 4, 4)
        return self.net(h)


class VAE(nn.Module):
    def __init__(self, in_channels=conf.CHANNELS, base=conf.VAE_BASE_CHANNELS, latent_dim=conf.VAE_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(in_channels, base, latent_dim)
        self.decoder = Decoder(in_channels, base, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device=None):
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


def vae_loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
             kl_weight: float = conf.VAE_KL_WEIGHT):
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    # mean over batch AND over latent dims -> O(1) scale, comparable to recon_loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    model = VAE()
    x = torch.randn(4, conf.CHANNELS, conf.IMAGE_SIZE, conf.IMAGE_SIZE)
    recon, mu, logvar = model(x)
    loss, rl, kl = vae_loss(recon, x, mu, logvar)
    print("recon:", recon.shape, "mu:", mu.shape, "logvar:", logvar.shape)
    print("loss:", loss.item(), "recon_loss:", rl.item(), "kl_loss:", kl.item())
    print("params:", sum(p.numel() for p in model.parameters()))
