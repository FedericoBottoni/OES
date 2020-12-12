import math
import numpy as np
from collections import namedtuple
import torch
from src.transfer.RND.Curiosity import Curiosity 

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'confidence'))

class PTL():
    def __init__(self, enable_transfer, n_instances, input_size, alpha, transfer_hyperparams):
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
        self._TRANSFER_DISC = transfer_hyperparams['TRANSFER_DISC']
        self._curiosity = [Curiosity(input_size, alpha, transfer_hyperparams['ENCODED_SIZE'])] * n_instances


    def update_state_visits(self, p, state):
        return -1 * self._curiosity[p].update(state)

    def confidence(self, p, state):
        return -1 * self._curiosity[p].uncertainty(state).item()

    # "provide" transfer
    def provide_transitions(self, p, replay_memory):
        confidences = torch.tensor(list(map(lambda x: x.confidence, replay_memory.memory)))
        indxs = self.provide_transitions_visits_top(confidences)
        sending_knowledge = [replay_memory.memory[i] for i in indxs]
        return sending_knowledge

    def provide_transitions_visits_top(self, memory):
        _, indxs = torch.topk(memory, self._TRANSFER_SIZE)
        return indxs

    def provide_transfer(self, replay_memory, policy_net):
        transitions = list()
        for p in range(self._n_instances):
            if len(replay_memory[p]) < self._TRANSFER_SIZE:
                    transitions.append([])
            else:
                # transitions.append(replay_memory[p].memory[-self._TRANSFER_SIZE:]) # BL
                transitions.append(self.provide_transitions(p, replay_memory[p])) # EXP1
        return transitions


    # "gather" transfer
    def gather_transitions(self, p, transfer_memory, size):
        confidences = torch.tensor(list(map(lambda x: x.confidence, transfer_memory.memory)))
        indxs = self.gather_transitions_visits_bottom(confidences, size)
        transitions = [transfer_memory.memory[i] for i in indxs]
        return transitions

    def gather_transitions_visits_bottom(self, buffer, k):
        _, indxs = torch.topk(buffer, k, largest=False)
        return indxs

    def gather_transfer(self, p, transfer_memory, size):
        # transitions = transfer_memory.sample(size) # BL
        transitions = self.gather_transitions(p, transfer_memory, size) # EXP2
        return transitions


    # Parameter functions
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

    def get_senders(self):
        return [n for n in range(self._n_instances) if n%2==0]
    
    def get_receivers(self):
        return [n for n in range(self._n_instances) if n%2==1]

    def get_active_receivers(self, procs_done):
        rec = self.get_receivers()
        return [rec[n] for n in range(len(rec)) if not procs_done[rec[n]]==1]
            
    # sender even, receiver odd
    def get_sender(self, receiver):
        if receiver % 2 == 0:
            return None
        else:
            return receiver - 1
    
    def get_receiver(self, sender): 
        if sender % 2 == 0:
            return sender + 1
        else:
            return None