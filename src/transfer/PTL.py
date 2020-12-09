import numpy as np
from collections import namedtuple
import gym
import torch
import random
import math

QVal = namedtuple('QVal', ('state', 'action', 'q'))

class PTL():
    def __init__(self, discretizer, n_instances, transfer_hyperparams, gym_environment, get_state_from_obs, c_plot):
        if n_instances <= 1:
            raise Exception('instance number cannot be less then 2')
        self._c_plot = c_plot
        self._n_instances = n_instances
        self._discretizer = discretizer
        self._CONFIDENCE = transfer_hyperparams['CONFIDENCE']
        self._TRANSFER_SIZE = transfer_hyperparams['TRANSFER_SIZE']
        self._THETA_MAX = transfer_hyperparams['THETA_MAX']
        self._THETA_MIN = transfer_hyperparams['THETA_MIN']
        self._THETA_DECAY = transfer_hyperparams['THETA_DECAY']
        self.elegibility = [False] * n_instances
        self._state_visits = [None] * n_instances

        self._virtual_env = gym.make(gym_environment).unwrapped
        self._virtual_env.reset()
        self._get_state_from_obs = get_state_from_obs

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
        self._THETA = self._THETA_MIN + (self._THETA_MAX - self._THETA_MIN) * \
                math.exp(-1. * steps_done / self._THETA_DECAY)
        if steps_done % 1000 == 0:
            print(self._THETA)

    def update_state_visits(self, p, state):
        st_index = tuple(self.get_index_from_state(state))
        self._state_visits[p][st_index] += 1
        if not self.elegibility[p]:
            self.elegibility[p] = self._state_visits[p][st_index] >= self._CONFIDENCE

    def merge(self, sending_p, received_qs, local_qs):
        states = [st[0] for st in received_qs]
        out_qs = list()
        for i_st in range(len(states)):
            st_index = tuple(self.get_index_from_state(states[i_st]))
            out_qs.append(received_qs[i_st].q + (self._state_visits[sending_p][st_index] / self._CONFIDENCE) * (local_qs[i_st].q - received_qs[i_st].q))
        return out_qs

    def select_senders(self):
        return [i for i, x in enumerate(self.elegibility) if x and self.allow_transfer()]

    def select_receivers(self, sender):
        return [i for i in range(self._n_instances) if i != sender and self.allow_transfer()]

    def transfer(self, policy_nets, step):
        senders = self.select_senders()
        sender_sizes = {}
        receiver_sizes = {}
        for i in range(self._n_instances):
            sender_sizes[str(i)] = 0
            receiver_sizes[str(i)] = 0
        out_data = list()
        for sender in senders:
            receivers = self.select_receivers(sender)
            sending_data = self.select_q_subtable(sender, policy_nets[sender])

            for receiver in range(len(receivers)):
                # local_data = self.sample_q_subtable(receiver, policy_nets[receiver], sending_data)
                # out_data[receiver] = self.merge(sender, sending_data, local_data)
                out, out_len = self.parse_out_data(receiver, sending_data)
                out_data.append((receiver, out))

                sender_sizes[str(sender)] += out_len
                receiver_sizes[str(receiver)] += out_len
        # self._c_plot.push_sending_dict(sender_sizes)
        # self._c_plot.push_receiving_dict(receiver_sizes)
        self.compute_theta_decay(step)
        return out_data
    
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
    
    def parse_out_data(self, receiver, sending_data):
        out = list()
        for qv in sending_data:
            observation, reward, _, _ = self._virtual_env.step(qv.action)
            out.append((qv.state, torch.tensor([[qv.action]]), \
                torch.from_numpy(self._get_state_from_obs(observation)).float(), torch.tensor([reward])))
        return out, len(out)