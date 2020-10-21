from torch.utils.tensorboard import SummaryWriter

class TBoard(object):

    def __init__(self, ):
        self._writer = SummaryWriter()
        self._n_loss = 0
        self._n_cm_reward = 0
        self._n_episode_len = 0
        self._n_value = 0

    def push(self, label, n_step, dataset):
        for n_iter in range(len(dataset)):
            self._writer.add_scalar(label, dataset[n_iter], n_iter + n_step)
            print(n_iter, " ", n_step, " ", dataset[n_iter])
    
    def push_loss(self, dataset):
        self.push("Loss", self._n_loss, dataset)
        self._n_loss += len(dataset)

    def push_cm_reward(self, dataset):
        self.push("Cumulative Reward", self._n_cm_reward, dataset)
        self._n_cm_reward += len(dataset)


    def push_episode_len(self, dataset):
        self.push("Episode Length", self._n_episode_len, dataset)
        self._n_episode_len += len(dataset)
    

    def push_value(self, dataset):
        self.push("value", self._n_value, dataset)
        self._n_value += len(dataset)

