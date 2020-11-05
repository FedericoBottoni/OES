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

        all_states = self._discretizer.get_bins()
        state_index_dicts = [] * n_instances
        for p in range(n_instances):
            state_index_dict = []
            for i_dim_states in range(len(all_states)):
                state_index_dict.append({})
                for i_st in range(len(all_states[i_dim_states])):
                    state_index_dict[i_dim_states][all_states[i_dim_states][i_st]] = i_st
            state_index_dicts[p] = state_index_dict

        self._state_index_dicts = state_index_dicts
        self._states = torch.cartesian_prod(torch.tensor(all_states[0]), torch.tensor(all_states[1])).detach()
        self._state_visits = [None] * n_instances
        for p in range(n_instances):
            self._state_visits[p] = np.ndarray(shape=tuple(self._discretizer.size), dtype=int)

    def get_state_index(self, p, state):
        len_state = len(state)
        out = [] * len_state
        for i_st in range(len_state):
             out[i_st] = self._state_index_dicts[p][i_st][self._discretizer.disc(state[i_st])]
        return out

    def allow_transfer(self):
        return random.random() <= self._THETA

    def compute_theta_decay(self, steps_done):
        self._THETA = self._THETA_END + (self._THETA_START - self._THETA_END) * \
                math.exp(-1. * steps_done / self._THETA_DECAY)

    def update_state_visits(self, p, state):
        st_index = self.get_state_index(p, state)
        self._state_visits[p][st_index] += 1
        if not self.elegibility[p]:
            self.elegibility[p] = self._state_visits[p][st_index] >= self._CONFIDENCE

    def merge(self, sending_p, received_qs, local_qs):
        states = [st[0] for st in received_qs]
        return received_qs + (self._state_visits[sending_p][states] / self._NUM_CV) * (local_qs - received_qs)

    def select_senders(self):
        print([i for i, x in enumerate(self.elegibility) if x])
        return [i for i, x in enumerate(self.elegibility) if x and self.allow_transfer()]

    def select_receivers(self, sender):
        procs = list(range(self._n_instances))
        return procs[procs != sender]

    def transfer(self, policy_nets, step):
        senders = self.select_senders()
        for sender in senders:
            print("transfering start")
            receivers = self.select_receivers(sender)
            
            sending_knowledge = self.select_q_subtable(sender, policy_nets[sender])

            merged_knowledge = [None] *  self._n_instances
            for receiver in range(receivers):
                local_knowledge = self.sample_q_subtable(receiver, policy_nets[receiver], sending_knowledge)
                merged_knowledge[receiver] = self.merge(sender, sending_knowledge, local_knowledge)
            
            print("transfering end")
        self.compute_theta_decay(step)
    
    def sample_q_subtable(self, p, policy_net, qvals):
        extracted_states = []
        actions = []
        for st, act in qvals:
            extracted_states.append(st)
            actions.append(act)
        values = policy_net(torch.tensor(extracted_states))
        qs = values.gather(1, torch.tensor(actions).view([-1, 1])).detach()
        out_know = []
        for i in range(len(self._states)):
            kn = QVal(self._states[i], actions[i], qs[i])
            out_know.append(kn)
        return out_know

    def select_q_subtable(self, p, policy_net):
        selected_states = self.select_knowledge_visits()
        values = policy_net(selected_states)
        actions = values.max(1)[1]
        qs = values.gather(1, actions.view([-1, 1])).detach()
        selected_i_qs = self.select_knowledge(qs)
        sending_knowledge = [None] * selected_i_qs.size()
        for i_state in selected_i_qs:
            sending_knowledge[i_state] = QVal(self._states[i_state], actions[i_state], qs[i_state])
        return sending_knowledge

    def select_knowledge_visits(self):
        conf_i_states = zip(*np.where(self._state_visits[p] >= self._CONFIDENCE))
        conf_states = list()
        for _ in range(len(conf_i_states)):
           conf_states.append([self.get_state_index(p, list(x)) for x in conf_i_states])
        return torch.tensor(conf_states)

    def select_knowledge(self, qs):
        return torch.topk(qs, self._TRANSFER_SIZE)[1].tolist()
