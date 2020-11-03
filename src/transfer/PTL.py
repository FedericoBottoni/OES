import numpy as np
from collections import namedtuple
import torch
import math

QVal = namedtuple('QVal', ('state', 'action', 'q'))
StateVisits = namedtuple('StateVisits', ('state', 'visits'))

class PTL():
    def __init__(self, discretizer, n_instances, transfer_size, confidence, numCV):
        if n_instances <= 1:
            raise Exception('instance number cannot be less then 2')
        self._n_instances = n_instances
        self._discretizer = discretizer
        self._transfer_size = transfer_size
        self._confidence = confidence
        self._numCV = numCV
        self.elegibility = [False] * n_instances

        all_states = self._discretizer.get_bins()
        self._states = torch.cartesian_prod(torch.tensor(all_states[0]), torch.tensor(all_states[1])).detach()
        self._state_visits = [list()] * n_instances
        for p in range(n_instances):
            for st in self._states:
                self._state_visits[p].append(StateVisits(st, 0))

    def update_state_visits(self, p, state):
        st_index = [x for x, y in enumerate(self._state_visits[p]) if y.state == state][0]
        if not self.elegibility[p]:
            self.elegibility[p] = self._state_visits[p][st_index] >= self._confidence

    
    def get_top_k(self, p, k):
        TODO

    def select_knowledge(self, p, data):
        topk = torch.topk(self._state_visits[p], k = self._transfer_size)
        mask = topk[0] >= self._confidence
        states = topk[1][torch.nonzero(mask)].numpy()
        sending_knowledge = [None] * states.size()
        for i_state in range(states):
            sending_knowledge[i_state] = (states[i_state], data[states[i_state]])
        return sending_knowledge

    def merge(self, sending_p, received_qs, local_qs):
        states = [st[0] for st in received_qs]
        return received_qs + (self._state_visits[sending_p][states] / self._numCV) * (local_qs - received_qs)

    def select_senders(self):
        return [i for i, x in enumerate(self.elegibility) if x]

    def select_receivers(self, sender):
        procs = list(range(self._n_instances))
        return procs[procs != sender]

    def transfer(self, policy_net):
        senders = self.select_senders()
        for sender in senders:
            receivers = self.select_receivers(sender,  self._n_instances)
            
            sending_knowledge = self.sample_q_table(policy_net[sender])
            sel_data = self.select_knowledge(sending_knowledge)

            merged_knowledge = [None] *  self._n_instances
            for receiver in range(receivers):
                local_knowledge = self.sample_q_table(policy_net[receiver], states=sending_knowledge)
                merged_knowledge[receiver] = self.merge(sender, sel_data, local_knowledge)

    def sample_q_table(self, policy_net, states=None):
            if states is None:
                values = policy_net(self._states)
                actions = values.max(0)[1]
            else:
                extracted_states = []
                actions = []
                for st, act in states:
                    extracted_states.append(st)
                    actions.append(act)
                values = policy_net(torch.tensor(extracted_states))
            qs = values.gather(1, actions).detach()
            out_know = []
            for i in range(self._states):
                kn = QVal(self._states[i], actions[i], qs[i])
                out_know.append(kn)
            return out_know