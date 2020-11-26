import numpy as np

class EarlyStopping(object):
    def __init__(self, n_instances, stop_reward_threshold, stop_range):
        self._enabled_module = not stop_reward_threshold is None and not stop_range is None
        if self._enabled_module:
            self._stop_reward_threshold = stop_reward_threshold
            self._stop_range = stop_range
            self._cm_rews = np.empty((stop_range, n_instances), dtype=np.float)
            self._i = np.zeros(n_instances, dtype=np.int)
        
    def eval_stop_condition(self, p, cm_reward):
        stop = False
        if self._enabled_module:
            self._cm_rews[self._i[p] % self._stop_range, p] = cm_reward
            if self._i[p] >= self._stop_range and self._cm_rews[:, p].mean() >= self._stop_reward_threshold:
                stop = True
            self._i[p] += 1
        return stop
       

    def on_stop(self, p):
        if self._enabled_module:
            print('Early stopping, condition reached by env#', p, 'after', self._i[p], 'episodes')