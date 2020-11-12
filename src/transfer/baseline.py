import math


class PTL():
    def __init__(self, enable_transfer, n_instances, transfer_hyperparams):
        if n_instances <= 1:
            raise Exception('instance number cannot be less then 2')
        self._n_instances = n_instances
        self._enable_transfer = enable_transfer
        self._TRANSFER_DELAY = transfer_hyperparams['TRANSFER_DELAY']
        self._TRANSFER_SIZE = transfer_hyperparams['TRANSFER_SIZE']
        self._THETA_START = transfer_hyperparams['THETA_START']
        self._THETA_END = transfer_hyperparams['THETA_END']
        self._THETA_DECAY = transfer_hyperparams['THETA_DECAY']

    def transfer(self, replay_memory):
        transitions = list()
        for p in range(self._n_instances):
            if len(replay_memory[p]) < self._TRANSFER_SIZE:
                    transitions.append([])
            else:
                transitions.append(replay_memory[p].memory[-self._TRANSFER_SIZE:])
        return transitions

    def get_theta(self, steps_done):
        if self._enable_transfer:
            return self._THETA_END + (self._THETA_START - self._THETA_END) * \
                math.exp(-1. * (steps_done - self._TRANSFER_DELAY) / self._THETA_DECAY)
        else:
            return 0
    
    def get_receiver(self, sender):
        if sender % 2 == 0:
            return sender + 1
        else:
            return sender - 1