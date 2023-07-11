import numpy as np
from skimage import draw
import copy
import math

class CanvasState:
    def __init__(self, width=256, height=256, minBrushHeight=-30, maxBrushHeight=10):
        self.width = width # width of canvas (in pixels)
        self.height = height # heigh of canvas (in pixels)
        self.brushX = width/2
        self.brushY = height/2
        
        # brush height 0 is level with the canvas
        # lower = thicker line, higher = thinner line or no line at all
        self.minBrushHeight = minBrushHeight
        self.maxBrushHeight = maxBrushHeight
        self.brushHeight = 0
        
        self.canvas = np.full((height, width), 255, dtype=np.uint8)

def getNextState(state, action):
    # state: a CanvasState
    # action: a 3d direction to move the brush (dx, dy, dh)
    # returns a new CanvasState with updated canvas and brush positions
    
    res = copy.deepcopy(state)
    
    # Get updated brush position
    newBrushX = state.brushX + action[0]
    newBrushY = state.brushY + action[1]
    newBrushHeight = max(state.minBrushHeight, min(state.brushHeight + action[2], state.maxBrushHeight))
    
    # Update canvas
    if newBrushHeight < 0:
        rr, cc = draw.disk(
            (res.height-newBrushY, newBrushX), # center
            radius=-newBrushHeight, # radius
            shape=(res.height, res.width)
        )
        res.canvas[rr, cc] = 0
    
    # Update brush positions
    res.brushX = newBrushX
    res.brushY = newBrushY
    res.brushHeight = newBrushHeight
    return res

def getCanvas(state):
    # just the base convas as a H x W x 3 array
    return np.stack([state.canvas, state.canvas, state.canvas], axis=2)

def getRenderedState(state):
    # canvas with current brush position drawn
    canvas = getCanvas(state)
    rr, cc, intensities = draw.circle_perimeter_aa(
        round(state.height-state.brushY), # row
        round(state.brushX), # column
        radius=round(abs(state.brushHeight)), # radius
        shape=(state.height, state.width)
    )
    
    for channel in range(3):
        canvas[rr, cc, channel] = canvas[rr, cc, channel] * (1-intensities)
    
    if state.brushHeight < 0:
        # draw in green
        canvas[rr, cc, 1] = canvas[rr, cc, 1] + 255*intensities
    else:
        # draw in red
        canvas[rr, cc, 0] = canvas[rr, cc, 0] + 255*intensities
    return canvas
