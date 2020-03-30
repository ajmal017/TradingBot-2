import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Qnet_linear(nn.Module):
    """ Network that learns Q function
		inputs: state vector
		outputs: action vector
    """

    def __init__(self, state_size, action_size, seed, fc1_unit=64,
                 fc2_unit = 64):

        super(Qnet_linear,self).__init__()

        self.seed = torch.manmual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_unit)
        seed.fc2 = nn.Linear(fc1_unit,fc2_unit)
        seed.fc3 = nn.Linear(fc2_unit,action_size)
        
    def forward(self, x):
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Qnet_LSTM(nn.Module):
	""" Alternate Q network """

    def __init__(self, state_size, hidden_size, action_size, num_layers=2):

        super(Qnet_LSTM, self).__init__()

        self.num_layers = num_layers
        self.lstm = nn.LSTM(state_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x):

        x = self.lstm(x)
        print(x.shape)
        # x = self.(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        # return tag_scores
