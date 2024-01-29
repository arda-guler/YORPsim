import numpy as np

class Sol:
    def __init__(self):
        self.pos = np.array([0, 0, 0])
        self.vel = np.array([0, 0, 0])
        self.luminosity = 3.828e26 # W
        self.mu = 1.3271244e20
