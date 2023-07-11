from canvas import *
import numpy as np

class CubicBezierGenerator:
    def gen(self):
        state = CanvasState()
        
        length = np.random.uniform(low=15.0, high=90.0, size=1)[0] # length of brush stroke
        height = np.random.uniform(low=-10.0, high=-5.0, size=1)[0] # height of brush at middle of stroke
        bend = np.random.uniform(low=-40.0, high=40.0, size=1)[0] # how far control points are from line between endpoints
        
        # generate p0, p1, p2, p3 starting as if from (0,0) and going to (length,0)
        p0 = np.array([0.0,0.0])
        p1 = np.array([length/3,bend])
        p2 = np.array([2*length/3,bend])
        p3 = np.array([length,0])
        
        # start stroke at (width/2, height/2)
        T = np.array([state.width/2, state.height/2])
        p0 += T
        p1 += T
        p2 += T
        p3 += T
        
        # draw the stroke
        def get_pos(t):
            # (x,y) coordinates
            pos = (1-t)**3 * p0 +\
                  3*(1-t)**2*t * p1 +\
                  3*(1-t)*t**2 * p2 +\
                  t**3 * p3
            
            # height
            if t < 1/10:
                h = t*10 * height
            elif 1/10 <= t and t < 2/3:
                h = height
            else:
                h = (1-t)*3 * height
            
            return np.array([pos[0], pos[1], h])
        
        cur = np.array([state.brushX, state.brushY, state.brushHeight])
        trajectory = [cur]
        
        t = 0
        max_dt = 0.05
        while t < 0.99:
            dt = max_dt
            while True:
                new_pos = get_pos(t + dt)
                action = new_pos - cur
                if abs(action[0]) > 1 or abs(action[1]) > 1 or abs(action[2]) > 0.2:
                    dt /= 2
                else:
                    t += dt
                    break
            state = getNextState(state, action)
            cur = new_pos
            trajectory.append(cur)
        
        return getCanvas(state)[:,:,0]/255, trajectory

class CircleGenerator:
    def gen(self):
        state = CanvasState()
        
        radius = np.random.uniform(low=5.0, high=10, size=1)[0] # radius of brush trajectory
        x_distortion = np.random.uniform(low=0.8, high=2.0, size=1)[0] # x scaling factor
        y_distortion = np.random.uniform(low=0.8, high=2.0, size=1)[0] # y scaling factor
        height = np.random.uniform(low=-10.0, high=-5.0, size=1)[0] # height of brush at middle of stroke
        
        # draw the stroke
        def get_pos(t):
            # (x,y) coordinates
            theta = 2*np.pi*t + np.pi
            pos = [state.width/2 + radius*x_distortion + x_distortion*radius*np.cos(theta),
                   state.height/2 + y_distortion*radius*np.sin(theta)]
            
            # height
            if t < 1/10:
                h = t*10 * height
            elif 1/10 <= t and t < 6/7:
                h = height
            else:
                h = (1-t)*7 * height
            
            return np.array([pos[0], pos[1], h])
        
        cur = np.array([state.brushX, state.brushY, state.brushHeight])
        trajectory = [cur]
        
        t = 0
        max_dt = 0.05
        while t < 0.99:
            dt = max_dt
            while True:
                new_pos = get_pos(t + dt)
                action = new_pos - cur
                if abs(action[0]) > 1 or abs(action[1]) > 1 or abs(action[2]) > 0.2:
                    dt /= 2
                else:
                    t += dt
                    break
            state = getNextState(state, action)
            cur = new_pos
            trajectory.append(cur)
        
        return getCanvas(state)[:,:,0]/255, trajectory
    