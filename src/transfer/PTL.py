import numpy as np
from collections import namedtuple
import itertools
import torch
import random
import math

QVal = namedtuple('QVal', ('state', 'action', 'q'))

class PTL():
    def __init__(self, discretizer, n_instances, transfer_hyperparams):
        if n_instances <= 1:
            raise Exception('instance number cannot be less then 2')
        self._n_instances = n_instances
        self._discretizer = discretizer
        self._CONFIDENCE = transfer_hyperparams['CONFIDENCE']
        self._NUM_CV = transfer_hyperparams['NUM_CV']
        self._TRANSFER_SIZE = transfer_hyperparams['TRANSFER_SIZE']
        self._THETA_START = transfer_hyperparams['THETA_START']
        self._THETA_END = transfer_hyperparams['THETA_END']
        self._THETA_DECAY = transfer_hyperparams['THETA_DECAY']
        self.elegibility = [False] * n_instances
        self._state_visits = [None] * n_instances
        
        for p in range(n_instances):
            self._state_visits[p] = np.zeros(shape=tuple(self._discretizer.size), dtype=int)

    def get_index_from_state(self, state):
        return self._discretizer.disc_index(state)

    def get_state_from_index(self, index):
        state = list()
        states = self._discretizer.get_bins()
        for i in range(len(index)):
            state.append(states[i][index[i]])
        return state

    def allow_transfer(self):
        return random.random() <= self._THETA

    def compute_theta_decay(self, steps_done):
        self._THETA = self._THETA_END + (self._THETA_START - self._THETA_END) * \
                math.exp(-1. * steps_done / self._THETA_DECAY)

    def update_state_visits(self, p, state):
        st_index = [[st] for st in self.get_index_from_state(state)] 
        self._state_visits[p][st_index] += 1
        if not self.elegibility[p]:
            self.elegibility[p] = (self._state_visits[p][st_index] >= self._CONFIDENCE)[0]

    def merge(self, sending_p, received_qs, local_qs):
        states = [st[0] for st in received_qs]
        out_qs = list()
        for i_st in range(len(states)):
            st_index = [[st] for st in self.get_index_from_state(states[i_st])]
            out_qs.append(received_qs[i_st].q + (self._state_visits[sending_p][st_index] / self._NUM_CV) * (local_qs[i_st].q - received_qs[i_st].q))
        return out_qs

    def select_senders(self):
        return [i for i, x in enumerate(self.elegibility) if x and self.allow_transfer()]

    def select_receivers(self, sender):
        procs = list(range(self._n_instances))
        return procs[procs != sender]

    def transfer(self, policy_nets, step):
        senders = self.select_senders()
        for sender in senders:
            #print("transfering start")
            receivers = self.select_receivers(sender)
            
            sending_knowledge = self.select_q_subtable(sender, policy_nets[sender])

            merged_knowledge = [None] *  self._n_instances
            for receiver in range(receivers):
                local_knowledge = self.sample_q_subtable(receiver, policy_nets[receiver], sending_knowledge)
                #print('sending_knowledge', sending_knowledge)
                #print('local_knowledge', local_knowledge)
                merged_knowledge[receiver] = self.merge(sender, sending_knowledge, local_knowledge)
            
            #print("transfering end")
        self.compute_theta_decay(step)
    
    def sample_q_subtable(self, p, policy_net, qvals):
        states = []
        actions = []
        for qval in qvals:
            states.append(qval.state.tolist())
            actions.append(qval.action)
        states = torch.tensor(states)
        values = policy_net(states)
        qs = values.gather(1, torch.tensor(actions).view([-1, 1])).detach()
        out_know = []
        for i in range(len(qvals)):
            kn = QVal(states[i], actions[i], qs[i].item())
            out_know.append(kn)
        return out_know

    def select_q_subtable(self, p, policy_net):
        selected_states = self.select_knowledge_visits(p)
        values = policy_net(selected_states)
        actions = values.max(1)[1]
        qs = values.gather(1, actions.view([-1, 1])).detach()[0]
        selected_i_qs = self.select_knowledge(qs)
        sending_knowledge = [None] * len(selected_i_qs)
        for i_state in selected_i_qs:
            sending_knowledge[i_state] = QVal(selected_states[i_state], actions[i_state].item(), qs[i_state].item())
        return sending_knowledge

    def select_knowledge_visits(self, p):
        conf_i_states = list(zip(*np.where(self._state_visits[p] >= self._CONFIDENCE)))
        conf_states = list()
        for i_conf_i_states in range(len(conf_i_states)):
            conf_states.append(self.get_state_from_index(list(conf_i_states[i_conf_i_states])))
        return torch.tensor(conf_states)

    def select_knowledge(self, qs):
        len_qs = len(qs)
        k = len_qs if len_qs < self._TRANSFER_SIZE else self._TRANSFER_SIZE
        return torch.topk(qs, k)[1].tolist()
