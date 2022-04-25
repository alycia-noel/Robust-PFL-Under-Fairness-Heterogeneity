from collections import OrderedDict
import torch
from torch import nn

class NNHyper(nn.Module):
    def __init__(self, n_nodes, embedding_dim, hidden_size, hnet_hidden_dim = 100, hnet_n_hidden=3):
        super().__init__()

        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hnet_hidden_dim
        self.n_hidden = hnet_n_hidden
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(num_embeddings=self.n_nodes, embedding_dim=self.embedding_dim)

        layers = [nn.Linear(self.embedding_dim, self.hidden_dim)]

        for _ in range(self.n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.mlp = nn.Sequential(*layers)

        self.fc1_weights = nn.Linear(self.hidden_dim, 2 * self.context_vector_size * self.hidden_size)
        self.fc1_bias = nn.Linear(self.hidden_dim, self.hidden_size)
        self.fc2_weights = nn.Linear(self.hidden_dim, self.hidden_size * self.hidden_size)
        self.fc2_bias = nn.Linear(self.hidden_dim, self.hidden_size)
        self.fc3_weights = nn.Linear(self.hidden_dim, 1 * self.hidden_size)
        self.fc3_bias = nn.Linear(self.hidden_dim, 1)

    def forward(self, idx):
        emd = self.embeddings(idx)

        features = self.mlp(emd)

        weights = OrderedDict({
            "fc1.weight": self.fc1_weights(features).view(self.hidden_size, 2*self.context_vector_size),
            "fc1.bias": self.fc1_bias(features).view(-1),
            "fc2.weight": self.fc2_weights(features).view(self.hidden_size, self.hidden_size),
            "fc2.bias": self.fc2_bias(features).view(-1),
            "fc3.weight": self.fc3_weights(features).view(1, self.hidden_size),
            "fc3.bias": self.fc3_bias(features).view(-1),
        })

        return weights

class LRHyper(nn.Module):
    def __init__(self, n_nodes, embedding_dim, hidden_size, hnet_hidden_dim = 100, hnet_n_hidden=3):
        super().__init__()

        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hnet_hidden_dim
        self.n_hidden = hnet_n_hidden
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(num_embeddings=self.n_nodes, embedding_dim=self.embedding_dim)

        layers = [nn.Linear(self.embedding_dim, self.hidden_dim)]

        for _ in range(self.n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.mlp = nn.Sequential(*layers)

        self.fc1_weights = nn.Linear(self.hidden_dim, self.hidden_size)
        self.fc1_bias = nn.Linear(self.hidden_dim, 1)

    def forward(self, idx):
        emd = self.embeddings(idx)

        features = self.mlp(emd)

        weights = OrderedDict({
            "fc1.weight": self.fc1_weights(features).view(1, self.hidden_size),
            "fc1.bias": self.fc1_bias(features).view(-1),
        })

        return weights

class LR_Context(nn.Module):
    def __init__(self, input_size, nn_hidden_size):
        super(LR_Context, self).__init__()

        self.input_size = input_size
        self.nn_hidden_size = nn_hidden_size

        # Neural Network
        self.fc1 = nn.Linear(self.input_size, 1)

    def forward(self, x):
        prediction = self.fc1(x)

        return prediction

class NN_Context(nn.Module):
    def __init__(self, input_size, nn_hidden_size, dropout):
        super(NN_Context, self).__init__()

        self.input_size = input_size
        self.nn_hidden_size = nn_hidden_size
        self.dropout = dropout


        # Neural Network
        self.fc1 = nn.Linear(self.input_size, self.nn_hidden_size)
        self.fc2 = nn.Linear(self.nn_hidden_size, self.nn_hidden_size)
        self.fc3 = nn.Linear(self.nn_hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x):
        prediction = self.dropout(self.relu(self.fc1(x)))
        prediction = self.dropout(self.relu(self.fc2(prediction)))
        prediction = self.fc3(prediction)

        return prediction

