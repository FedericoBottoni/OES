import torch.nn as nn

class TargetNet(nn.Module):

    def __init__(self, input_size, output_len):
        super(TargetNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_len)
        )

    def forward(self, x):
        return self.layers(x)

class PredictorNet(nn.Module):

    def __init__(self, input_size, output_len):
        super(PredictorNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_len)
        )

    def forward(self, x):
        return self.layers(x)
