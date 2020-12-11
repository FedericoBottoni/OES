import torch.optim as optim
import torch.nn as nn
from src.transfer.RND.Nets import PredictorNet, TargetNet

class Curiosity:
    def __init__(self, input_size, alpha, encoded_size):
        self._target = TargetNet(input_size, encoded_size)
        self._predictor = PredictorNet(input_size, encoded_size)
        self._optimizer = optim.RMSprop(self._predictor.parameters(), lr=alpha)
        self._criterion = nn.MSELoss()
    
    def uncertainty(self, state):
        predicted = self._predictor(state)
        target = self._target(state)

        return self._criterion(predicted, target)
        

    def update(self, state):
        self._optimizer.zero_grad() 

        predicted = self._predictor(state)
        target = self._target(state)

        loss = self._criterion(predicted, target)

        loss.backward()
        self._optimizer.step()