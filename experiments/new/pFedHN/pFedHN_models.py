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
    def __init__(self, input_size, bound, fairness):
        super(LR, self).__init__()

        self.num_classes = 2
        self.fairness = fairness
        self.bound = 1 / bound
        self.eps = bound
        self.sensitive_classes = [0, 1]
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 1)
        self.y_classes = [0, 1]

        if self.fairness == 'dp':
            self.n_constraints = 2 * self.num_classes
            self.dim_condition = self.num_classes + 1
            self.M = torch.zeros((self.n_constraints, self.dim_condition))
            self.c = torch.tensor([self.eps for x in range(self.n_constraints)])

            for i in range(self.n_constraints):
                j = i % 2

                if i == 0 or i == 1:
                    if j == 0:
                        self.M[i, j] = 1.0
                        self.M[i, -1] = -1.0
                    else:
                        self.M[i, j - 1] = -1.0
                        self.M[i, -1] = 1.0
                elif i == 2 or i == 3:
                    if j == 0:
                        self.M[i,j+1] = 1.0
                        self.M[i, -1] = -1
                    else:
                        self.M[i, j] = -1
                        self.M[i, -1] = 1

        elif self.fairness == 'eo':
            self.n_constraints = self.num_classes * self.num_classes * 2
            self.dim_condition = self.num_classes * (self.num_classes + 1)
            self.M = torch.zeros((self.n_constraints, self.dim_condition))
            self.c = torch.tensor([self.eps for x in range(self.n_constraints)])

            for i in range(self.n_constraints):
                j = i % 4

                if i == 0 or i == 1 or i == 2 or i == 3:
                    if j == 0:
                        self.M[i, j] = 1.0
                        self.M[i, -2] = -1
                        self.M[i, -1] = -1
                    elif j == 1:
                        self.M[i, j - 1] = -1.0
                        self.M[i, -2] = 1
                        self.M[i, -1] = 1
                    elif j == 2:
                        self.M[i, j - 1] = 1.0
                        self.M[i, -2] = -1
                        self.M[i, -1] = -1
                    elif j == 3:
                        self.M[i, j - 2] = -1.0
                        self.M[i, -2] = 1
                        self.M[i, -1] = 1
                if i == 4 or i == 5 or i == 6 or i == 7:
                    if j == 0:
                        self.M[i, j + 2] = 1.0
                        self.M[i, -2] = -1
                        self.M[i, -1] = -1
                    elif j == 1:
                        self.M[i, j + 1] = -1.0
                        self.M[i, -2] = 1
                        self.M[i, -1] = 1
                    elif j == 2:
                        self.M[i, j + 1] = 1.0
                        self.M[i, -2] = -1
                        self.M[i, -1] = -1
                    elif j == 3:
                        self.M[i, j] = -1.0
                        self.M[i, -2] = 1
                        self.M[i, -1] = 1

    def mu_f(self, out, sensitive, y):
        expected_values_list = []
        if self.fairness == 'eo':
            compare = torch.tensor([]).to(sensitive.device)

            for u in self.sensitive_classes:
                for v in self.y_classes:
                    idx_true = (y == v) * (sensitive == u)

                    if torch.equal(out[idx_true], compare):
                        expected_values_list.append(out.mean() * 0)
                    else:
                        expected_values_list.append(out[idx_true].mean())

            for v in self.y_classes:
                idx_true = y == v
                #print('this is broken:', torch.sum(idx_true.type(torch.FloatTensor)))
                if torch.sum(idx_true.type(torch.FloatTensor)) == 0:
                    print(torch.sum(idx_true.type(torch.FloatTensor)))
                    exit(1)
                    expected_values_list.append(out.mean() * 0)
                else:
                    expected_values_list.append(out[idx_true].mean())

        elif self.fairness == 'dp':
            sensitive = sensitive.view(out.shape)

            compare = torch.tensor([]).to(sensitive.device)
            for v in self.sensitive_classes:
                idx_true = sensitive == v

                if torch.equal(out[idx_true], compare):
                    expected_values_list.append(out.mean()*0)
                else:
                    expected_values_list.append(out[idx_true].mean())
            expected_values_list.append(out.mean())

        return torch.stack(expected_values_list)

    def M_mu_q(self, pred, sensitive, y):
        const = torch.mv(self.M.to(pred.device), self.mu_f(pred, sensitive, y)) - self.c.to(pred.device)
        #print(self.mu_f(pred, sensitive, y))
        #print(const)
        return const

    def forward(self, x, s, y):
        prediction = torch.sigmoid(self.fc1(x))

        if self.fairness == 'none':
            m_mu_q = None
        else:
            m_mu_q = self.M_mu_q(prediction, s, y)

        return prediction, m_mu_q

class Constraint(torch.nn.Module):
    def __init__(self, fair, bound):
        super().__init__()
        self.bound = 1 / bound
        self.fair = fair
        if self.fair == 'dp':
            self.register_parameter(name='lmbda', param=torch.nn.Parameter(torch.rand((4,1))))
        elif self.fair == 'eo':
            self.register_parameter(name='lmbda', param=torch.nn.Parameter(torch.rand((8,1))))

    def forward(self, value):
        with torch.no_grad():
            if np.linalg.norm(self.lmbda.data.cpu().numpy()) > self.bound:
                minimum = self.lmbda.data.min()
                for i in range(len(self.lmbda.data)):
                    self.lmbda.data[i] = minimum
                if np.linalg.norm(self.lmbda.data.cpu().numpy()) > self.bound:
                    print('hit over bound')
                    exit(1)

            self.lmbda.data = torch.clamp(self.lmbda.data, min=0)

        for i, lm in enumerate(self.lmbda.data):
            if lm.item() < 0:
                print('hit', lm.item())
                exit(1)

        loss = torch.matmul(self.lmbda.T, value)

        return loss

