import torch
import numpy as np
from skimage import draw
from canvas import *
from torchvision import transforms

class TrajectoryVisualizer():
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.G = 512
        self.resize64 = transforms.Resize((64, 64), antialias=True)

    def visualize_trajectory(self, trajectory):
        # trajectory: N x 3
        trajectory = torch.concat([torch.Tensor([[0.0, 0.0, 0.0]]), trajectory], dim=0) # starting point of trajectory
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
    
    def render_trajectory(self, trajectory):
        trajectory = (trajectory - np.array([-1, -1, 0])) * np.array([128, 128, -30])
        def is_action_valid(action):
            return abs(action[0]) < 1 and abs(action[1]) < 1 and abs(action[2]) < 1
        
        state = CanvasState()
        for dest in trajectory:
            start = torch.Tensor([state.brushX, state.brushY, state.brushHeight])
            cur = start.clone()
            t = 0
            max_dt = 0.05
            while not is_action_valid(dest - cur):
                dt = min(max_dt, 1-t)
                while True:
                    new_pos = start * (1-(t + dt)) + dest * (t + dt)
                    action = new_pos - cur
                    if is_action_valid(action):
                        t += dt
                        break
                    dt /= 2
                state = getNextState(state, action)
                cur += action
            state = getNextState(state, dest - cur)
        return self.resize64(torch.Tensor(getCanvas(state)[:,:,0]/255).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
