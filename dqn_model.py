import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, hidden_layer_size) -> None:
        super().__init__()
        # we are starting with a tuple of 4 tetris features
        self.fc1 = nn.Linear(4, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        # final layer will just output an estimated state value (raw)
        self.out = nn.Linear(hidden_layer_size, 1)

    def forward(self, features):
        # pass data through layers
        # we are using rectified linear units for intermediate layers
        inputs = features
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        inputs = F.relu(self.fc3(inputs))
        
        # we don't use an activation function for the output layer
        # we will just use raw outputs as estimated state value
        return self.out(inputs)

