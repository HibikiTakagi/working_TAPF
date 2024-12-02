import gymnasium as gym
from gymnasium import spaces

class Environment(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def step(self, action):
        pass
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
    
    def seed(self, seed=None):
        pass

def State(Hashable):
    pass

def Action(Hashable):
    pass
