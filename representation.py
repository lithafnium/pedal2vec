"""
John Mayer guitar --> disjoint latent space --> John Mayer guitar + clean guitar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    # TODO: learn a separate thread of layers for audio representation
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        reconstruction = torch.sigmoid(self.fc3(h))
        return reconstruction


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# Loss Function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Example usage
input_dim = 1024  # This should match your audio input size
latent_dim = 64  # Size of the latent space
vae = VAE(input_dim, latent_dim)


def main():
    # Assuming 'data' is your input audio tensor
    recon_batch, mu, logvar = vae(data)
    loss = loss_function(recon_batch, data, mu, logvar)


if __name__ == "__main__":
    main()
