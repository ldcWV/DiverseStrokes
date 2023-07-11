from torch.utils.data import Dataset

class TrajDataset(Dataset):
    def __init__(self, trajectories):
        # trajectories: N x 8 x 4
        self.trajectories = trajectories
    
    def __len__(self):
        return self.trajectories.shape[0]
    
    def __getitem__(self, idx):
        return self.trajectories[idx]