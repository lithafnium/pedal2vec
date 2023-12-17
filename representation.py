"""
John Mayer guitar --> disjoint latent space --> John Mayer guitar + clean guitar
"""

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


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


def split_audio(
    raw_audio: np.ndarray, sampling_rate: int = 44100, chunk_size: int = 100
):
    """
    Wright et. al: the training data was split into 100 ms training examples
    """

    chunk = sampling_rate * (chunk_size / 1000)
    cutoff = len(raw_audio) % int(chunk)
    raw_audio = raw_audio[:-cutoff]
    audio_splits = np.split(raw_audio, chunk)
    np.random.shuffle(audio_splits)
    audio_splits = torch.tensor(audio_splits, dtype=torch.float32)
    return audio_splits


def train(train_tensor, validation_tensor):
    train_dataset = TensorDataset(train_tensor)
    validation_dataset = TensorDataset(validation_tensor)

    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


def main():
    # with wave.open('wavFiles/GuitarMixIn.wav', 'r') as wav_file:
    #     # Get parameters
    #     n_channels, sampwidth, framerate, n_frames, comptype, compname = wav_file.getparams()
    #     print(wav_file.getparams())

    #     # Read audio frames
    #     frames = wav_file.readframes(n_frames)

    #     # Convert frames to byte array
    #     frames = np.frombuffer(frames, dtype=np.int16)

    # Load wav as mono at 44.1 kHz sampling rate
    # Normalized to [-1, 1]
    y, sr = librosa.load("isolated-guitar/Nirvana-Teen-Spirit.wav", sr=44100, mono=True)
    # print(y)
    # print(y.shape)
    # y, sr = librosa.load("wavFiles/GuitarMixIn.wav", sr=44100, mono=True)
    # print(y)
    # print(y.shape)

    # print("Frames?")
    # for f in frames:
    #     print(f)
    # Example usage
    audio_set = split_audio(y)
    print(audio_set)
    data = audio_set[0]

    train_ratio = 0.8
    split_index = int(len(audio_set) * train_ratio)
    print(split_index)
    train_data = audio_set[:split_index]
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    validation_data = audio_set[split_index:]
    validation_tensor = torch.tensor(validation_data, dtype=torch.float32)

    # Number of audio samples in each training example
    input_dim = len(audio_set[0])
    latent_dim = 64
    vae = VAE(input_dim, latent_dim)
    recon_batch, mu, logvar = vae(data)
    loss = loss_function(recon_batch, data, mu, logvar)


if __name__ == "__main__":
    main()
