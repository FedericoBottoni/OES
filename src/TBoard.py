# tensorboard --logdir=runs

from torch.utils.tensorboard import SummaryWriter

class TBoard(object):

    # tb_activated = False is a tensorboard mock
    def __init__(self, tb_activated):
        self._tb_activated = tb_activated
        self._writer = SummaryWriter() if self._tb_activated else None
        self._n_loss = 0
        self._n_cm_reward = 0
        self._n_cm_reward_ep = 0
        self._n_episode_len = 0
        self._n_value = 0

    def push(self, label, n_step, dataset):
        if(self._tb_activated):
            for n_iter in range(len(dataset)):
                self._writer.add_scalar(label, dataset[n_iter], n_iter + n_step)
    
    def push_loss(self, dataset):
        self.push("Loss", self._n_loss, dataset)
        self._n_loss += len(dataset)

    def push_cm_reward(self, dataset):
        self.push("Reward/Cumulative Reward", self._n_cm_reward, dataset)
        self._n_cm_reward += len(dataset)

    def push_cm_reward_ep(self, dataset):
        self.push("Reward/Cumulative Reward Ep.", self._n_cm_reward_ep, dataset)
        self._n_cm_reward_ep += len(dataset)

    def push_episode_len(self, dataset):
        self.push("Episode/Episode Length", self._n_episode_len, dataset)
        self._n_episode_len += len(dataset)
    

    def push_value(self, dataset):
        self.push("Reward/Value", self._n_value, dataset)
        self._n_value += len(dataset)

