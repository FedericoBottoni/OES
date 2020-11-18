import math
import numpy as np
from collections import namedtuple
import gym
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class PTL():
    def __init__(self, enable_transfer, n_instances, gym_environment, discretizer, transfer_hyperparams):
        if n_instances <= 1:
            raise Exception('instance number cannot be less then 2')
        self._n_instances = n_instances
        self._enable_transfer = enable_transfer
        self._discretizer = discretizer
        self._TRANSFER_APEX = transfer_hyperparams['TRANSFER_APEX']
        self._TRANSFER_SIZE = transfer_hyperparams['TRANSFER_SIZE']
        self._THETA_MAX = transfer_hyperparams['THETA_MAX']
        self._THETA_MIN = transfer_hyperparams['THETA_MIN']
        self._THETA_DECAY = transfer_hyperparams['THETA_DECAY']
        self._THETA_DIFF = transfer_hyperparams['THETA_MAX'] - transfer_hyperparams['THETA_MIN']
        self._TRANSFER_DISC = transfer_hyperparams['TRANSFER_DISC']
        self._states = self._discretizer.get_bins()
        self._state_visits = [None] * n_instances
    
        self._virtual_env = gym.make(gym_environment).unwrapped
        self._virtual_env.reset()

        for p in range(n_instances):
            self._state_visits[p] = torch.zeros(tuple(self._discretizer.size), dtype=int)
        
    def get_index_from_state(self, state):
        return self._discretizer.disc_index(state)

    def get_state_from_index(self, index):
        state = list()
        for i in range(len(index)):
            state.append(self._states[i][index[i]])
        return state
    
    def update_state_visits(self, p, state):
        st_index = tuple(self.get_index_from_state(state))
        self._state_visits[p][st_index] += 1


    # "provide" transfer
    def provide_transitions(self, p, policy_net):
        selected_states_i = self.provide_transitions_visits_top(p)
        selected_states = torch.tensor(list(map(self.get_state_from_index, selected_states_i)))
        values = policy_net(selected_states)
        actions = values.max(1)[1]
        sending_knowledge = []
        for i_state in range(len(selected_states)):
            self._virtual_env.set_state(tuple(selected_states[i_state].numpy()))
            observation, reward, _, _ = self._virtual_env.step(actions[i_state].item())
            sending_knowledge.append(Transition( \
                selected_states[i_state], torch.tensor([[actions[i_state]]]), \
                    torch.from_numpy(observation).float(), torch.tensor([reward])))
        return sending_knowledge

    def provide_transitions_visits_top(self, p):
        n_states = self._TRANSFER_DISC ** len(self._states)
        k = min(n_states, self._TRANSFER_SIZE)
        _, indxs_flt = torch.topk(self._state_visits[p].flatten(), k)
        indxs = np.array(np.unravel_index(indxs_flt.numpy(), self._state_visits[p].shape)).T
        return indxs

    def provide_transfer(self, replay_memory, policy_net):
        transitions = list()
        for p in range(self._n_instances):
            if len(replay_memory[p]) < self._TRANSFER_SIZE:
                    transitions.append([])
            else:
                transitions.append(replay_memory[p].memory[-self._TRANSFER_SIZE:]) # BL
                #transitions.append(self.provide_transitions(p, policy_net[p])) # EXP1
        return transitions


    # "gather" transfer
    def gather_transitions(self, p, transfer_buffer, size):
        #print(self._state_visits[p])
        len_transfer_buffer = len(transfer_buffer)
        visits = [None] * len_transfer_buffer
        for i in range(len_transfer_buffer):
            states_index = self._discretizer.disc_index(transfer_buffer[i].state)
            visits[i] = self._state_visits[p][tuple(states_index)]
            #print(states_index, visits[i])
        tr_indexes = self.gather_transitions_visits_bottom(torch.tensor(visits), size)
        transitions = [transfer_buffer[i] for i in tr_indexes]
        return transitions

    def gather_transitions_visits_bottom(self, buffer, k):
        #print(buffer, k)
        _, indxs_flt = torch.topk(buffer, k, largest=False)
        return indxs_flt

    def gather_transfer(self, p, transfer_buffer, size):
        transitions = self.gather_transitions(p, transfer_buffer, size) # EXP2
        # transitions = transfer_memory[p].sample(transfer_batch_size) # BL
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