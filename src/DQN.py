import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, input_len, output_len):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_len)
        )


    def forward(self, x):
        return self.layers(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
