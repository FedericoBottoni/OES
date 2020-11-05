from Discretizer import Discretizer
import numpy as np
import torch

def gaussian(x, mu=0, sig=100):
    return np.exp(-np.power(x - 0, 2.) / (2 * np.power(sig, 2.)))

class CartPoleDiscretizer():
    def __init__(self, n_cart_position, n_cart_velocity, n_pole_angle, n_pole_angle_vel, function=None):
        self.cart_position_disc = Discretizer(-4.8, 4.8, n_cart_position, function=function)
        self.cart_velocity_disc = Discretizer(-1000, 1000, n_cart_velocity, function=gaussian)
        self.pole_angle_disc = Discretizer(-0.418, 0.418, n_pole_angle, function=function)
        self.pole_angle_vel_disc = Discretizer(-1000, 1000, n_pole_angle_vel, function=gaussian)
        

    def disc(self, state):
        return torch.tensor([self.cart_position_disc.disc(state[0]), \
            self.cart_velocity_disc.disc(state[1]), \
            self.pole_angle_disc.disc(state[2]), \
            self.pole_angle_vel_disc.disc(state[3])], dtype=torch.float)