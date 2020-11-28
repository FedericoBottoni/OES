import numpy as np

class Discretizer():
    def __init__(self, minimum, maximum, n_bins, function=None):
        self.min = minimum
        self.max = maximum
        self.n_bins = n_bins
        self.range = maximum - minimum
        
        _offset = self.range / n_bins
        _uniform_bins = [x * _offset + minimum for x in range(n_bins)]
        
        if function:
            bins = list(map(function, _uniform_bins))
        else:
            bins = _uniform_bins
        self.bins = bins

    def disc(self, nums):
        return self.bins[self.disc_index(nums)]

    def disc_index(self, nums):
        out = np.digitize(nums, self.bins, right=True)
        return out if out < self.n_bins else self.n_bins - 1