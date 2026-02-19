import torch
import torch.nn as nn
import torch.nn.functional as F

# input size is the four features we are getting from tetris engine
INPUT_SIZE = 4

# the deep learning approximator model we'll use
# it utilizes pytorch neural networks
# inputs -> the feautres we'll get from the tetris engine (tuple of 4)
# output -> an estimated value of a given tetris board state
class DQNModel(nn.Module):

    def __init__(self, hidden_layer_size: int = 64) -> None:
        super().__init__()
        # we will use 3 hidden layers, default is 64 nodes
        self.fc1 = nn.Linear(INPUT_SIZE, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        # output layer will output the raw score
        self.out = nn.Linear(hidden_layer_size, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            # if we fill the weights with the random values, the value for a state can explode
            # in the network, and our RL model can report extreme losses
            # to avoid the possibility of this spike, we initialize the NNs with semi 
            # orthogonal matrices 
            nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
            # we initialize the bias as 0
            nn.init.zeros_(layer.bias)
        
        # optimiziation for early training, we don't want our model to initially reach a very
        # negative or positive score on the first state with the random values
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, features):
        # CHANGE
        # pytorch layers expect inputs in batches
        # so features should have 2 dims, first one is batch
        # if tensor has only 1 dim add another dim at axis 0
        if features.dim() == 1:
            features = features.unsqueeze(0)
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # we don't use activation for output layer
        return self.out(x).squeeze(-1)

