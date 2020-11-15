# tensorboard --logdir=runs

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

SR_LABELS = ['Senders', 'Receivers']

def group_SR(groups, procs_done, data):
    sending = list()
    receiving = list()
    for i in range(len(data)):
        if i in groups[0] and not procs_done[i] == 1:
            sending.append(data[i])
        if i in groups[1] and not procs_done[i] == 1:
            receiving.append(data[i])
        dataset = list()
        labels = list()
        if len(sending) > 0:
            dataset.append(torch.tensor(sending, dtype=torch.float).mean().item())
            labels.append(SR_LABELS[0])
        if len(receiving) > 0:
            dataset.append(torch.tensor(receiving, dtype=torch.float).mean().item())
            labels.append(SR_LABELS[1])
    return dataset, labels

class TBoard(object):

    # tb_activated = False is a tensorboard mock
    def __init__(self, tb_activated, ptl, n_instances):
        self._tb_activated = tb_activated
        self._ptl = ptl
        self._instances = [str(i) for i in range(n_instances)]
        self._writer = SummaryWriter() if self._tb_activated else None
        self._n_loss = 0
        self._n_cm_reward = 0
        self._n_cm_reward_ep = 0
        self._n_episode_len = 0
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
    

    # Arrays
    def push_ar_loss(self, dones, dataset):
        self.push_scalars("Loss/Loss", self._instances, self._n_loss, dataset)
        if not self._ptl._enable_transfer:
            self.push_scalar("Loss/Loss Mn.", self._n_loss, dataset.mean())
        else:
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], dones, dataset)
            self.push_scalars("Loss/Loss Mn", labels, self._n_loss, g_dataset)
        self._n_loss += 1

    def push_ar_cm_reward(self, dones, dataset):
        self.push_scalars("Reward/Cumulative Reward", self._instances, self._n_cm_reward, dataset)
        if not self._ptl._enable_transfer:
            self.push_scalar("Reward/Cumulative Reward Mn.", self._n_cm_reward, dataset.mean())
        else:
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], dones, dataset)
            self.push_scalars("Reward/Cumulative Reward Mn", labels, self._n_cm_reward, g_dataset)
        self._n_cm_reward += 1

    def push_ar_cm_reward_ep(self, dones, dataset):
        self.push_scalars("Reward/Cumulative Reward Ep.", self._instances, self._n_cm_reward_ep, dataset)
        if not self._ptl._enable_transfer:
            self.push_scalar("Reward/Cumulative Reward Ep. Mn.", self._n_cm_reward_ep, dataset.mean())
        else:
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], dones, dataset)
            self.push_scalars("Reward/Cumulative Reward Ep. Mn", labels, self._n_cm_reward_ep, g_dataset)
        self._n_cm_reward_ep += 1

    def push_ar_episode_len(self, dones, dataset):
        self.push_scalars("Episode/Episode Length", self._instances, self._n_episode_len, dataset)
        if not self._ptl._enable_transfer:
            self.push_scalar("Episode/Episode Length Mn.", self._n_episode_len, dataset.mean())
        else:
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], dones, dataset)
            self.push_scalars("Episode/Episode Length Mn", labels, self._n_episode_len, g_dataset)
        self._n_episode_len += 1


    # Multi charts
    def push_sending_dict(self, my_dict):
        self.push_scalars_dict("Transfer/Sending", self._sending_data, my_dict)
        self._sending_data += 1

    def push_receiving_dict(self, my_dict):
        self.push_scalars_dict("Transfer/Receiving", self._receiving_data, my_dict)
        self._receiving_data += 1
        