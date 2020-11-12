# tensorboard --logdir=runs

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class TBoard(object):

    # tb_activated = False is a tensorboard mock
    def __init__(self, tb_activated, n_instances):
        self._tb_activated = tb_activated
        self._instances = [str(i) for i in range(n_instances)]
        self._writer = SummaryWriter() if self._tb_activated else None
        self._n_loss = 0
        self._n_cm_reward = 0
        self._n_cm_reward_ep = 0
        self._n_episode_len = 0
        self._q_value_mean = 0
        self._q_value_var = 0
        self._sending_data = 0
        self._receiving_data = 0

    def dispose(self):
        if self._tb_activated:
            self._writer.flush()
            self._writer.close()

    def push_scalar(self, tag, n_step, data):
        if(self._tb_activated):
            self._writer.add_scalar(tag, data, n_step)
    
    def push_scalars(self, tag, labels, n_step, datasets):
        if self._tb_activated:
            plotset = {}
            for i in range(len(labels)):
                plotset[labels[i]] = datasets[i]
            self._writer.add_scalars(tag, plotset, n_step)
    
    def push_scalars_dict(self, tag, n_step, plotset):
        if(self._tb_activated):
            self._writer.add_scalars(tag, plotset, n_step)
    

    # Scalars
    def push_loss(self, dataset):
        self.push_scalar("Loss/Loss Mn.", self._n_loss, dataset)
        self._n_loss += 1

    def push_cm_reward(self, dataset):
        self.push_scalar("Reward/Cumulative Reward Mn.", self._n_cm_reward, dataset)
        self._n_cm_reward += 1

    def push_cm_reward_ep(self, dataset):
        self.push_scalar("Reward/Cumulative Reward Ep. Mn.", self._n_cm_reward_ep, dataset)
        self._n_cm_reward_ep += 1

    def push_episode_len(self, dataset):
        self.push_scalar("Episode/Episode Length Mn.", self._n_episode_len, dataset)
        self._n_episode_len += 1


    # Arrays
    def push_ar_loss(self, dataset):
        self.push_scalars("Loss/Loss", self._instances, self._n_loss, dataset)
        self.push_loss(dataset.mean())

    def push_ar_cm_reward(self, dataset):
        self.push_scalars("Reward/Cumulative Reward", self._instances, self._n_cm_reward, dataset)
        self.push_cm_reward(dataset.mean())

    def push_ar_cm_reward_ep(self, dataset):
        self.push_scalars("Reward/Cumulative Reward Ep.", self._instances, self._n_cm_reward_ep, dataset)
        self.push_cm_reward_ep(dataset.mean())

    def push_ar_episode_len(self, dataset):
        self.push_scalars("Episode/Episode Length", self._instances, self._n_episode_len, dataset)
        self.push_episode_len(dataset.mean())



    # Multi charts
    def push_q_value_mean_dict(self, my_dict):
        self.push_scalars_dict("Reward/Q-Value Mean", self._q_value_mean, my_dict)
        self._q_value_mean += 1
    
    def push_q_value_var_dict(self, my_dict):
        self.push_scalars_dict("Reward/Q-Value Var", self._q_value_var, my_dict)
        self._q_value_var += 1
    
    def push_sending_dict(self, my_dict):
        self.push_scalars_dict("Transfer/Sending", self._sending_data, my_dict)
        self._sending_data += 1

    def push_receiving_dict(self, my_dict):
        self.push_scalars_dict("Transfer/Receiving", self._receiving_data, my_dict)
        self._receiving_data += 1

    def push_q_values(self, action_dict, state_action_values, action_batch):
        q_values = state_action_values.view(1, -1)[0]
        q_labels = action_batch.view(1, -1)[0]
        single_action_labels = torch.unique(q_labels)
        single_action_values_mean = {}
        single_action_values_var = {}
        for action_lbl in single_action_labels:
            a_mask = q_labels == action_lbl
            a_q_values = q_values[torch.nonzero(a_mask, as_tuple=True)].view(1, -1)[0]
            single_action_values_mean[action_dict[str(action_lbl.item())]] = a_q_values.mean().item()
            single_action_values_var[action_dict[str(action_lbl.item())]] = a_q_values.var().item()
        self.push_q_value_mean_dict(single_action_values_mean)
        self.push_q_value_var_dict(single_action_values_var)