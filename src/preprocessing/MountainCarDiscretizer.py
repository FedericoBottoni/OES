from Discretizer import Discretizer

class MountainCarDiscretizer():
    def __init__(self, env, n_position_bins, n_speed_bins, function=None):
        self.position_disc = Discretizer(env.min_position, env.max_position, n_position_bins, function=function)
        self.speed_disc = Discretizer(0, env.max_speed, n_speed_bins, function=function)
        

    def disc(self, position, speed):
        return [self.position_disc.disc(position), self.speed_disc.disc(speed)]