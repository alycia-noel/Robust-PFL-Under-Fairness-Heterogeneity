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
        out = torch.sigmoid(y)

        return out, y


class NN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 10, 10], dropout_rate=.45):
        super(NN, self).__init__()
        self.input_size = input_size
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(self.input_size, hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc3 = nn.Linear(hidden_sizes[2], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[2])
        self.fc5 = nn.Linear(hidden_sizes[2], 1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, data):
        x1 = F.relu(self.fc1(data))
        x2 = self.dropout(x1)
        x3 = self.relu(self.fc2(x2))
        x4 = self.dropout(x3)
        x5 = self.relu(self.fc3(x4))
        x6 = self.dropout(x5)
        x7 = self.relu(self.fc4(x6))
        x8 = self.dropout(x7)
        x9 = self.fc5(x8)
        out = torch.sigmoid(x9)

        return out, x9