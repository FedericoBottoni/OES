import math


class PTL():
    def __init__(self, enable_transfer, n_instances, transfer_hyperparams):
        if n_instances <= 1:
            raise Exception('instance number cannot be less then 2')
        self._n_instances = n_instances
        self._enable_transfer = enable_transfer
        self._TRANSFER_APEX = transfer_hyperparams['TRANSFER_APEX']
        self._TRANSFER_SIZE = transfer_hyperparams['TRANSFER_SIZE']
        self._THETA_MAX = transfer_hyperparams['THETA_MAX']
        self._THETA_MIN = transfer_hyperparams['THETA_MIN']
        self._THETA_DECAY = transfer_hyperparams['THETA_DECAY']
        self._THETA_DIFF = transfer_hyperparams['THETA_MAX'] - transfer_hyperparams['THETA_MIN']

    def transfer(self, replay_memory):
        transitions = list()
        for p in range(self._n_instances):
            if len(replay_memory[p]) < self._TRANSFER_SIZE:
                    transitions.append([])
            else:
                transitions.append(replay_memory[p].memory[-self._TRANSFER_SIZE:])
        return transitions

    def get_theta(self, episode):
        if self._enable_transfer:
            if episode <= self._TRANSFER_APEX:
                theta = ((self._THETA_DIFF / self._TRANSFER_APEX ** 2) * episode ** 2) + self._THETA_MIN
            else:
                theta = self._THETA_MIN + self._THETA_DIFF * \
                    math.exp(-1. * (episode - self._TRANSFER_APEX) / self._THETA_DECAY)
            return theta
        else:
            return 0
    
    def get_receiver(self, sender):
        if sender % 2 == 0:
            return sender + 1
        else:
            return sender - 1