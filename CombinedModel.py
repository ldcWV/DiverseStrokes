import torch
from torch import nn
from TrajVAE import TrajVAE
from TrajToStroke import TrajToStroke

class CombinedModel(nn.Module):
    def __init__(self, traj_vae, traj_to_stroke, device):
        super(CombinedModel, self).__init__()
        self.device = device
        self.traj_vae = traj_vae.to(device)
        self.traj_to_stroke = traj_to_stroke.to(device)
    
    def forward(self, x):
        # x: batch_size x 5
        traj = self.traj_vae.decode(x)
        stroke = self.traj_to_stroke(traj)
        return stroke, traj
    
    def sample_latent(self, num_samples):
        max_batch = 16
        latents = []
        strokes = []
        trajectories = []
        for i in range(0, num_samples, max_batch):
            k = min(num_samples-i, max_batch)
            z = torch.randn(k, self.traj_vae.latent_dim).to(self.device)
            s, t = self.forward(z)
            latents.append(z)
            strokes.append(s)
            trajectories.append(t)
        latents = torch.cat(latents, dim=0)
        strokes = torch.cat(strokes, dim=0)
        trajectories = torch.cat(trajectories, dim=0)
        return latents, strokes, trajectories
    