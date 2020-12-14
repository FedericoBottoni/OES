# tensorboard --logdir=runs

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

SR_LABELS = ['Senders', 'Receivers']

def group_SR(groups, data, n, procs_done=None):
    sending = list()
    receiving = list()
    dataset = list()
    labels = list()
    i_undone = 0
    for i in range(n):
        if procs_done is None or not procs_done[i] == 1:
            if i in groups[0]:
                sending.append(data[i_undone])
            if i in groups[1]:
                receiving.append(data[i_undone])
            i_undone += 1
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
        self._n_instances = n_instances
        self._writer = SummaryWriter() if self._tb_activated else None
        self._n_step = 0
        self._n_episode = 0
        self._sending_data = 0
        self._receiving_data = 0

    def add_step(self):
        self._n_step += 1

    def add_episode(self):
        self._n_episode += 1

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

    def get_async_undone_data(self, dones, dataset):
        filt_labels = [self._instances[i] for i in range(len(self._instances)) if dones[i] == 0]
        filt_dataset = [dataset[i] for i in range(len(dataset)) if dones[i] == 0]
        return filt_labels, filt_dataset
    
    def get_sync_undone_data(self, dataset):
        filt_labels = list()
        filt_dataset = list()
        dones = list()
        for i in range(len(dataset)):
            done = 1
            if not dataset[i] is None:
                filt_labels.append(self._instances[i])
                filt_dataset.append(dataset[i])
                done = 0
            dones.append(done)
        return filt_labels, filt_dataset, dones

    # Arrays
    def push_ar_loss(self, dones, dataset):
        filt_labels, filt_dataset = self.get_async_undone_data(dones, dataset)
        self.push_scalars("Loss/Loss", filt_labels, self._n_step, filt_dataset)
        if not self._ptl._enable_transfer:
            self.push_scalar("Loss/Loss Mn", self._n_step, torch.tensor(dataset, dtype=torch.float).mean())
        else:
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], dataset, \
                self._n_instances, procs_done=dones)
            self.push_scalars("Loss/Loss Mn", labels, self._n_step, g_dataset)

    def push_ar_cm_reward(self, dones, dataset):
        filt_labels, filt_dataset = self.get_async_undone_data(dones, dataset)
        self.push_scalars("Reward/Cumulative Reward", filt_labels, self._n_step, filt_dataset)
        if not self._ptl._enable_transfer:
            self.push_scalar("Reward/Cumulative Reward Mn", self._n_step, torch.tensor(dataset, dtype=torch.float).mean())
        else:
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], dataset, \
                self._n_instances, procs_done=dones)
            self.push_scalars("Reward/Cumulative Reward Mn", labels, self._n_step, g_dataset)

    def push_ar_cm_reward_ep(self, dataset):
        filt_labels, filt_dataset, dones = self.get_sync_undone_data(dataset)
        self.push_scalars("Reward/Cumulative Reward Ep", filt_labels, self._n_episode, filt_dataset)
        if not self._ptl._enable_transfer:
            self.push_scalar("Reward/Cumulative Reward Ep Mn", self._n_episode, torch.tensor(filt_dataset, dtype=torch.float).mean())
        else:
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], filt_dataset, \
                self._n_instances, procs_done=dones)
            self.push_scalars("Reward/Cumulative Reward Ep Mn", labels, self._n_episode, g_dataset)

    def push_ar_episode_len(self, dataset):
        filt_labels, filt_dataset, dones = self.get_sync_undone_data(dataset)
        self.push_scalars("Episode/Episode Length", filt_labels, self._n_episode, filt_dataset)
        if not self._ptl._enable_transfer:
            self.push_scalar("Episode/Episode Length Mn", self._n_episode, torch.tensor(filt_dataset, dtype=torch.float).mean())
        else:
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], filt_dataset, \
                self._n_instances, procs_done=dones)
            self.push_scalars("Episode/Episode Length Mn", labels, self._n_episode, g_dataset)

    def push_ar_confidence(self, dataset):
        filt_labels, filt_dataset, dones = self.get_sync_undone_data(dataset)
        self.push_scalars("Transfer/Confidence AVG Ep", filt_labels, self._n_episode, filt_dataset)
        if not self._ptl._enable_transfer:
            self.push_scalar("Transfer/Confidence AVG Ep Mn", self._n_episode, torch.tensor(filt_dataset, dtype=torch.float).mean())
        else:
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], filt_dataset, \
                self._n_instances, procs_done=dones)
            self.push_scalars("Transfer/Confidence AVG Ep Mn", labels, self._n_episode, g_dataset)

    def push_sending(self, dones, dataset):
        if self._ptl._enable_transfer:
            filt_labels, filt_dataset = self.get_async_undone_data(dones, dataset)
            self.push_scalars("Transfer/Sending", filt_labels, self._n_step, filt_dataset)
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], filt_dataset, \
                self._n_instances, procs_done=dones)
            self.push_scalars("Transfer/Sending Mn", labels, self._n_step, g_dataset)

    def push_receiving(self, dones, dataset):
        if self._ptl._enable_transfer:
            filt_labels, filt_dataset = self.get_async_undone_data(dones, dataset)
            self.push_scalars("Transfer/Receiving", filt_labels, self._n_step, filt_dataset)
            g_dataset, labels = group_SR([self._ptl.get_senders(), self._ptl.get_receivers()], filt_dataset, \
                self._n_instances, procs_done=dones)
            self.push_scalars("Transfer/Receiving Mn", labels, self._n_step, g_dataset)
        