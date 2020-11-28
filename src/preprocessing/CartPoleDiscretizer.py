from src.preprocessing.Discretizer import Discretizer
import numpy as np
import torch

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class CartPoleDiscretizer():
    def __init__(self, env, n_state_bins, function=None):
        env_gaussian = lambda x: gaussian(x, 0, 3)
        self.cart_position_disc = Discretizer(-4.8, 4.8, n_state_bins[0], function=function)
        self.cart_velocity_disc = Discretizer(-2, 2, n_state_bins[1], function=function)
        self.pole_angle_disc = Discretizer(-0.418, 0.418, n_state_bins[2], function=function)
        self.pole_angle_vel_disc = Discretizer(-2, 2, n_state_bins[3], function=function)
        self.size = n_state_bins
        
    def get_bins(self):
        return [self.cart_position_disc.bins, self.cart_velocity_disc.bins, \
            self.pole_angle_disc.bins, self.pole_angle_vel_disc.bins]

    def disc(self, state):
        return torch.tensor([ \
            self.cart_position_disc.disc(state[0].item()), \
            self.cart_velocity_disc.disc(state[1].item()), \
            self.pole_angle_disc.disc(state[2].item()), \
            self.pole_angle_vel_disc.disc(state[3].item())], dtype=torch.float)
    
    def disc_index(self, state):
        return [ \
            self.cart_position_disc.disc_index(state[0].item()), \
            self.cart_velocity_disc.disc_index(state[1].item()), \
            self.pole_angle_disc.disc_index(state[2].item()), \
            self.pole_angle_vel_disc.disc_index(state[3].item())]