from torch.utils.data import Dataset

class StrokeTrajDataset(Dataset):
    def __init__(self, strokes, trajectories):
        # strokes: N x 64 x 64
        # trajectories: N x 8 x 4
        assert strokes.shape[0] == trajectories.shape[0]
        self.strokes = strokes.unsqueeze(1)
        self.trajectories = trajectories
    
    def __len__(self):
        return self.strokes.shape[0]
    
    def __getitem__(self, idx):
        return self.strokes[idx], self.trajectories[idx]