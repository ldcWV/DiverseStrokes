from torch import nn
from torch.nn import functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.hidden_fc = nn.Sequential(
            nn.Linear(24, hidden_dim),
            nn.LeakyReLU()
        )
        self.mean_fc = nn.Linear(hidden_dim, latent_dim)
        self.var_fc = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, trajectory):
        x = torch.flatten(trajectory, start_dim=1)
        x = self.hidden_fc(x)
        mean = self.mean_fc(x)
        logvar = self.var_fc(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        
        self.hidden_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.decode_fc = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.hidden_fc(x)
        x = self.decode_fc(x)
        x = x.view(-1, 8, 3)
        # scale so that x is [0, 1], y is in [-1, 1], z is in [0, 1]
        device = self.dummy_param.device
        traj_scale = torch.Tensor([0.5, 1, 0.5]).to(device)
        traj_shift = torch.Tensor([0.5, 0, 0.5]).to(device)
        x = x*traj_scale + traj_shift
        return x

class TrajVAE(nn.Module): # inspired by https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    def __init__(self):
        super(TrajVAE, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
                
        self.hidden_dim = 128
        self.latent_dim = 5
        
        self.encoder = Encoder(self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.hidden_dim, self.latent_dim)
        
    def encode(self, trajectory):
        # trajectory: N x 8 x 3
        mean, logvar = self.encoder(trajectory)
        return mean, logvar
    
    def decode(self, x):
        return self.decoder(x)
    
    def sample(self, mean, logvar):
        var = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        return mean + eps*var
    
    def forward(self, trajectory):
        # Returns [decoder output, encoder input, log(variance), mean]
        mean, logvar = self.encode(trajectory)
        latent = self.sample(mean, logvar)
        trajectory_dec = self.decode(latent)
        return [trajectory_dec, trajectory, logvar, mean]
    
    def reconstruction_loss(self, args):
        trajectory_dec = args[0]
        trajectory = args[1]
        return F.mse_loss(trajectory_dec, trajectory)
    
    def kl_loss(self, args):
        logvar = args[2]
        mean = args[3]
        return torch.mean(-0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=0)
    
    def loss(self, args, beta, capacity):
        reconstruction_loss = self.reconstruction_loss(args)
        kl_loss = self.kl_loss(args)
        return reconstruction_loss + beta*(kl_loss - capacity).abs()
    
    def sample_latent(self, num_samples):
        device = self.dummy_param.device
        max_batch = 16
        latents = []
        trajectories = []
        for i in range(0, num_samples, max_batch):
            k = min(num_samples-i, max_batch)
            z = torch.randn(k, self.latent_dim).to(device)
            t = self.decode(z)
            latents.append(z)
            trajectories.append(t)
        latents = torch.cat(latents, dim=0)
        trajectories = torch.cat(trajectories, dim=0)
        return latents, trajectories
    