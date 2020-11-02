from Discretizer import Discretizer

class MountainCarDiscretizer():
    def __init__(self, env, n_state_bins, function=None):
        self.position_disc = Discretizer(env.min_position, env.max_position, n_state_bins[0], function=function)
        self.speed_disc = Discretizer(0, env.max_speed, n_state_bins[1], function=function)
        self.size = n_state_bins
        
    def get_bins(self):
        return [self.position_disc.bins, self.speed_disc.bins]

    def disc(self, position, speed):
        return [self.position_disc.disc(position), self.speed_disc.disc(speed)]