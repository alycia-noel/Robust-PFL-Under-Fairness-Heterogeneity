from collections import OrderedDict
import torch
from torch import nn
import numpy as np

class LRHyper(nn.Module):
    def __init__(self, device,n_nodes, embedding_dim, context_vector_size, hidden_size, hnet_hidden_dim = 100, hnet_n_hidden=3):
        super().__init__()

        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.context_vector_size = context_vector_size
        self.hidden_dim = hnet_hidden_dim
        self.n_hidden = hnet_n_hidden
        self.hidden_size = hidden_size
        self.device = device
        self.embeddings = nn.Embedding(num_embeddings=self.n_nodes, embedding_dim=self.embedding_dim)

        layers = [nn.Linear(self.embedding_dim, self.hidden_dim)]

        for _ in range(self.n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.mlp = nn.Sequential(*layers)

        self.fc1_weights = nn.Linear(self.hidden_dim, self.context_vector_size)
        self.fc1_bias = nn.Linear(self.hidden_dim, 1)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "fc1.weight": self.fc1_weights(features).view(1, self.context_vector_size),
            "fc1.bias": self.fc1_bias(features).view(-1),
        })

        return weights

class LR(nn.Module):
    def __init__(self, input_size, bound):
        super(LR, self).__init__()

        self.bound = torch.Tensor([bound])
        self.sensitive_classes = [0, 1]
        self.input_size = input_size
        self.n_constraints = 4 # 2 * num classes
        self.dim_condition = 3 # num_classes + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.fc1 = nn.Linear(self.input_size, 1)

        for i in range(self.n_constraints):
            j = i % 2
            if j == 0:
                self.M[i, j] = 1.0
                self.M[i, -1] = -1.0
            else:
                self.M[i, j - 1] = -1.0
                self.M[i, -1] = 1.0

    def mu_q(self, pred, sensitive):
        sensitive = sensitive.view(pred.shape)

        expected_values_list = []
        for v in self.sensitive_classes:
            idx_true = sensitive == v

            if torch.sum(idx_true.type(torch.FloatTensor)) == 0:
                expected_values_list.append(pred.mean()*0)
            else:
                expected_values_list.append(pred[idx_true].mean())
        expected_values_list.append(pred.mean())

        return torch.stack(expected_values_list)

    def M_mu_q(self, pred, sensitive):
        return torch.mv(self.M.to(pred.device), self.mu_q(pred, sensitive) - self.bound.to(pred.device))

    def forward(self, x, s):
        prediction = torch.sigmoid(self.fc1(x))

        m_mu_q = self.M_mu_q(prediction, s)

        return prediction, m_mu_q

class Constraint(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.register_parameter(name='lmbda', param=torch.nn.Parameter(torch.rand((4,1))))

    def forward(self, value):

        with torch.no_grad():
            self.lmbda.data = self.lmbda.data.clamp(min=0, max=(np.linalg.norm(self.lmbda.data.cpu().numpy()) + (1/.05)))

        for i, lm in enumerate(self.lmbda.data):
            if lm.item() < 0:
                print('hit', lm.item())
                exit(1)

        loss = torch.matmul(self.lmbda.T, value)

        return loss
