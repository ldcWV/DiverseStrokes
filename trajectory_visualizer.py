import torch
import numpy as np
from skimage import draw

class TrajectoryVisualizer():
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.G = 512

    def visualize_trajectory(self, trajectory):
        # trajectory: N x 3
        path = torch.zeros(trajectory.shape)
        path[:, 0] = (trajectory[:, 0]-self.xmin)/(self.xmax - self.xmin) * self.G
        path[:, 1] = (trajectory[:, 1]-self.ymin)/(self.ymax - self.ymin) * self.G
        path[:, 2] = (trajectory[:, 2]-self.zmin)/(self.zmax - self.zmin) * self.G
        
        # canvas = torch.zeros((3, self.G, self.G))
        canvas = np.full((self.G, self.G), 255, dtype=np.uint8)
        
        # draw trajectory path
        for i in range(len(path)-1):
            x1, y1 = path[i][0], path[i][1]
            x2, y2 = path[i+1][0], path[i+1][1]
            radius = self.G/80
            color = torch.Tensor([0., 0., 1.])
            #canvas = tc.draw_line(x1, y1, x2, y2, radius, color, canvas)
            rr, cc = draw.polygon_perimeter([self.G-1 - y2, self.G-1 - y1, self.G-1 - y2], [x2, x1, x2], shape=(self.G, self.G)) # hack to draw line as degenerate polygon
            canvas[rr, cc] = 0
        
        # draw circles along trajectory
        for i in range(len(path)):
            x, y = path[i][0], path[i][1]
            radius = path[i][2]/40 + self.G/160
            color = torch.Tensor([0., 1., 0.])
            #canvas = tc.draw_circle(x, y, radius, color, canvas)
            rr, cc = draw.disk((self.G-1 - y, x), radius=radius, shape=(self.G, self.G))
            canvas[rr, cc] = 0
        
        return torch.from_numpy(canvas)
