import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class LR(nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        y = self.fc1(x)

        return y


class NN(nn.Module):
    def __init__(self, input_size, hidden_size=10, dropout_rate=.5):
        super(NN, self).__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, data):
        prediction = self.dropout(self.relu(self.fc1(data)))
        prediction = self.dropout(self.relu(self.fc2(prediction)))
        prediction = self.fc3(prediction)

        return prediction