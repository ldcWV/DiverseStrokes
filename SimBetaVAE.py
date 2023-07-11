from torch import nn
from torch.nn import functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, stroke_hidden_dims, traj_hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        encoder_layers = []
        prev_d = 1
        for d in stroke_hidden_dims:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=prev_d,
                          out_channels=d,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(d),
                nn.LeakyReLU()
            )
            encoder_layers.append(conv_block)
            prev_d = d
        self.stroke_encoder = nn.Sequential(*encoder_layers)
        self.traj_encoder = nn.Sequential(
            nn.Linear(24, traj_hidden_dim),
            nn.LeakyReLU()
        )
        self.mean_fc = nn.Linear(2*2*stroke_hidden_dims[-1] + traj_hidden_dim, latent_dim)
        self.var_fc = nn.Linear(2*2*stroke_hidden_dims[-1] + traj_hidden_dim, latent_dim)
    
    def forward(self, stroke, trajectory):
        x1 = self.stroke_encoder(stroke)
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(trajectory, start_dim=1)
        x2 = self.traj_encoder(x2)
        x = torch.cat([x1, x2], dim=1)
        
        mean = self.mean_fc(x)
        logvar = self.var_fc(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, stroke_hidden_dims, traj_hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        
        self.traj_hidden_dim = traj_hidden_dim
        self.dummy_param = nn.Parameter(torch.empty(0))
        
        self.decoder_fc = nn.Linear(latent_dim, 2*2*stroke_hidden_dims[-1] + traj_hidden_dim)
        decoder_layers = []
        stroke_hidden_dims.reverse()
        for i in range(len(stroke_hidden_dims) - 1):
            deconv_block = nn.Sequential(
                nn.ConvTranspose2d(stroke_hidden_dims[i],
                                   stroke_hidden_dims[i + 1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(stroke_hidden_dims[i + 1]),
                nn.LeakyReLU()
            )
            decoder_layers.append(deconv_block)
        self.stroke_decoder = nn.Sequential(*decoder_layers)
        self.stroke_decoder_final = nn.Sequential(
            nn.ConvTranspose2d(stroke_hidden_dims[-1],
                               stroke_hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(stroke_hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(stroke_hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.traj_decoder = nn.Sequential(
            nn.Linear(traj_hidden_dim, 24),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.decoder_fc(x)
        
        x1 = x[:,:-self.traj_hidden_dim]
        x1 = x1.view(-1, 128, 2, 2)
        x1 = self.stroke_decoder(x1)
        x1 = self.stroke_decoder_final(x1)
        
        x2 = x[:,-self.traj_hidden_dim:]
        x2 = self.traj_decoder(F.leaky_relu(x2))
        x2 = x2.view(-1, 8, 3)
        # scale so that x is [0, 1], y is in [-1, 1], z is in [0, 1]
        device = self.dummy_param.device
        traj_scale = torch.Tensor([0.5, 1, 0.5]).to(device)
        traj_shift = torch.Tensor([0.5, 0, 0.5]).to(device)
        x2 = x2*traj_scale + traj_shift
        
        return x1, x2

class SimBetaVAE(nn.Module): # inspired by https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    def __init__(self):
        super(SimBetaVAE, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
                
        stroke_hidden_dims = [8, 16, 32, 64, 128]
        self.traj_hidden_dim = 128
        self.latent_dim = 5
        
        self.encoder = Encoder(stroke_hidden_dims, self.traj_hidden_dim, self.latent_dim)
        self.decoder = Decoder(stroke_hidden_dims, self.traj_hidden_dim, self.latent_dim)
        
    def encode(self, stroke, trajectory):
        # stroke: N x 64 x 64
        # trajectory: N x 8 x 3
        mean, logvar = self.encoder(stroke, trajectory)
        return mean, logvar
    
    def decode(self, x):
        x1, x2 = self.decoder(x)
        return x1, x2
    
    def sample(self, mean, logvar):
        var = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mean)
        return mean + eps*var
    
    def forward(self, stroke, trajectory):
        # Returns [decoder output, encoder input, log(variance), mean]
        mean, logvar = self.encode(stroke, trajectory)
        latent = self.sample(mean, logvar)
        stroke_dec, trajectory_dec = self.decode(latent)
        return [(stroke_dec, trajectory_dec), (stroke, trajectory), logvar, mean]
    
    def reconstruction_loss(self, args):
        stroke_dec, trajectory_dec = args[0]
        stroke, trajectory = args[1]
        stroke_reconstruction_loss = F.mse_loss(stroke_dec, stroke)
        trajectory_reconstruction_loss = F.mse_loss(trajectory_dec, trajectory)
        return stroke_reconstruction_loss + 10*trajectory_reconstruction_loss
    
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
        strokes = []
        trajectories = []
        for i in range(0, num_samples, max_batch):
            k = min(num_samples-i, max_batch)
            z = torch.randn(num_samples, self.latent_dim).to(device)
            s, t = self.decode(z)
            latents.append(z)
            strokes.append(s)
            trajectories.append(t)
        latents = torch.cat(latents, dim=0)
        strokes = torch.cat(strokes, dim=0)
        trajectories = torch.cat(trajectories, dim=0)
        return latents, strokes, trajectories
    