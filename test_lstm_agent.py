import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_basics import MyBatchNorm1d, PositionalEncoding

d_net= 64
d_latent= 4
T= 200
N=525


class ResLSTM(nn.Module):
    """
    Generic LSTM-based sequence-to-sequence model
    """

    def __init__(self, d):
        super().__init__()
        d_net = d
        self.lstm1 = nn.LSTM(input_size=d_net, hidden_size=d_net//2, num_layers=2, dropout=0.1, bidirectional=True, batch_first=True)
        self.bn1 = MyBatchNorm1d(d_net)
        self.lstm2 = nn.LSTM(input_size=d_net, hidden_size=d_net//2, num_layers=2, dropout=0.1, bidirectional=True, batch_first=True)
        self.bn2 = MyBatchNorm1d(d_net)

    def forward(self, x): # dim: (B, T, d)
        x = self.bn1(x + self.lstm1(x)[0])
        x = self.bn2(x + self.lstm2(x)[0])
        return x # dim: (B, T, d)


# Short-cut function to make an MLP
def make_mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in ,d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))

class Spikes2Vec(nn.Module):
    """
    Simple deep learning model to transform a raster of shape (B, T, N) into a low dimensional latent of shape (B, d)
    """

    def __init__(self, N, d_net):
        super().__init__()
        self.proj_input = nn.Linear(N ,d_net)
        self.input_norm = MyBatchNorm1d(N)
        self.lstm = ResLSTM(d_net)
        self.bn = MyBatchNorm1d(d_net)

    def forward(self, z):
        z = self.proj_input(self.input_norm(z)) # dim: (B x T x d_net)

        # TODO:
        # Using the modules defined in init. Implement the Encoder:
        # Should be ~2 lines.

        ###BEGIN SOLUTION
        out = self.lstm(z) # out: (B x T x d_net)
        out = out.mean(1)
        ###END SOLUTION

        out = self.bn(out) # out: (B x d_net)
        return out

class Vec2Spikes(nn.Module):
    """
    Simple deep learning model to transform a low-dimensional latent of shape (B, d) into a firing probability or shape (B, T, N)
    """

    def __init__(self, d_latent, N, d=32, d_time=64):
        super().__init__()
        self.d = d
        self.N = N
        self.pe = PositionalEncoding(d_time, T=T)

        # Encode the latent variable
        self.proj_latent = nn.Linear(d_latent, d)
        self.proj_time = nn.Linear(d_time, d)

        self.lstm_encoder = ResLSTM(d)
        self.proj_encoder = nn.Linear(d ,N)

        self.bn_latent = MyBatchNorm1d(d)
        self.bn_time = MyBatchNorm1d(d)

    def forward(self, latent, t0, tend):
        B, d_latent = latent.shape
        # latent = self.bn_latent(self.proj_latent(latent)) # dim: B x d
        latent = self.proj_latent(latent) # dim: B x d

        time_latent = self.pe(t0, tend).to(latent.device)
        time_latent = self.bn_time(self.proj_time(time_latent)) # dim: T x d

        assert list(latent.shape) == [B, self.d]
        assert list(time_latent.shape) == [tend -t0, self.d]

        # TODO:
        # Using the modules defined in init. Finish the implementation of the decoder as described in the markdown above.
        # Should be ~3 lines

        ###BEGIN SOLUTION
        u_latent = F.relu(latent[: ,None] + time_latent[None, ...])
        u_latent = self.lstm_encoder(u_latent)
        u_latent = self.proj_encoder(u_latent) # dim:
        ###END SOLUTION

        return u_latent # dim: B x T x N

class VAE(nn.Module):
    """
    This is a spike recording VAE similar to LFADS
    https://arxiv.org/abs/1608.06315
    https://www.nature.com/articles/s41592-018-0109-9
    """

    def __init__(self, d_net=64, d_latent=2):
        super().__init__()
        self.d_net = d_net
        self.encoder = Spikes2Vec(N, d_net)
        self.decoder = Vec2Spikes(d_latent, N, d_net)

        self.latent_mean = make_mlp(d_net, d_net, d_latent)
        self.latent_var = make_mlp(d_net, d_net, d_latent)

    def encode(self, z):
        x = self.encoder(z)
        mu = self.latent_mean(x)
        logvar = self.latent_var(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        return mu + std * torch.randn_like(std)

    def decode(self, latent, t0=0, tend=T):
        return self.decoder(latent ,t0 ,tend)

    def forward(self, z_target, t0, tend):
        B, T, _ = z_target.shape
        mu, logvar = self.encode(z_target) # dim: B x d_latent (both)
        latent = self.reparametrize(mu, logvar) if self.training else mu # dim: B x d_latent

        loss_dkl = - 0.5 * (1 + logvar - mu.pow
            (2) - logvar.exp()) # VAE Prior: latent should be gaussian of mean 0 and Covariance I
        loss_dkl = loss_dkl.mean() # .clip(min=0.5).sum(-1).mean() # beta VAE clip trick to stabilize training
        u_decoded = self.decoder(latent ,t0 ,tend)
        # Binary spikes are, for instance, a Bernoulli with probability sigmoid(u)
        return u_decoded, latent, loss_dkl


class MyAgent:
        def __init__(self, ):
            self.vae = VAE(d_net=d_net, d_latent=d_latent)
            self.vae.load_state_dict(torch.load("trained_vae.pt", weights_only=False, map_location=torch.device("cpu")))
            self.vae.eval()

        def encode(self, X: np.ndarray):
            z = torch.tensor(X, dtype=torch.float)  # device is cpu device
            mu, logvar = self.vae.encode(z)
            return mu.detach().cpu().numpy()

        def decode(self, z: np.ndarray):
            latent = torch.tensor(z, dtype=torch.float)
            logits = self.vae.decode(latent)
            spikes = (torch.sigmoid(logits) > torch.rand_like(logits)).float()
            return spikes.detach().cpu().numpy()