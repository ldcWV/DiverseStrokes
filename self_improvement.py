from trajectory_visualizer import TrajectoryVisualizer
from torch.nn import functional as F
from torch.utils.data import Dataset
import torch

# class LatentStrokeTrajDataset(Dataset):
#     def __init__(self, latents, strokes, trajectories):
#         # strokes: N x 64 x 64
#         # trajectories: N x 8 x 4
#         assert latents.shape[0] == strokes.shape[0] == trajectories.shape[0]
#         self.latents = latents
#         self.strokes = strokes.unsqueeze(1)
#         self.trajectories = trajectories
    
#     def __len__(self):
#         return self.strokes.shape[0]
    
#     def __getitem__(self, idx):
#         return self.latents[idx], self.strokes[idx], self.trajectories[idx]

def evaluate_consistency(model, num_samples=128):
    _, strokes, trajectories = model.sample_latent(num_samples)
    tv = TrajectoryVisualizer(-1, 1, -1, 1, 0, 1)
    tot = 0
    for i in range(num_samples):
        stroke_true = torch.Tensor(tv.render_trajectory(trajectories[i].cpu().detach()))
        stroke_hat = strokes[i].cpu().detach().squeeze(0)
        tot += F.mse_loss(stroke_true, stroke_hat)
    return tot / num_samples

# def train_consistency(model, optimizer, epochs=100, num_samples=128):
#     latents, _, trajectories = model.sample_latent(num_samples)
#     tv = TrajectoryVisualizer(-1, 1, -1, 1, 0, 1)
#     strokes = []
#     for i in range(num_samples):
#         strokes.append(torch.from_numpy(tv.render_trajectory(trajectories[i])).float())
#     strokes = torch.cat(strokes, dim=0)
#     # todo: train on (latent, (stroke, trajectory)) pairs
