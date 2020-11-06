from Discretizer import Discretizer
import torch

class MountainCarDiscretizer():
    def __init__(self, env, n_state_bins, function=None):
        self.position_disc = Discretizer(env.min_position, env.max_position, n_state_bins[0], function=function)
        self.speed_disc = Discretizer(0, env.max_speed, n_state_bins[1], function=function)
        self.size = n_state_bins
        
    def get_bins(self):
        return [self.position_disc.bins, self.speed_disc.bins]

    def disc(self, state):
        return torch.tensor([self.position_disc.disc(state[0].item()), self.speed_disc.disc(state[1].item())], dtype=torch.float)
    
    def disc_index(self, state):
        return [self.position_disc.disc_index(state[0].item()), self.speed_disc.disc_index(state[1].item())]