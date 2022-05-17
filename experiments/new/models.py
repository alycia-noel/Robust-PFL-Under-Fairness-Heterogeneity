from collections import OrderedDict
import torch
from torch import nn
import numpy as np

class NNHyper(nn.Module):
    def __init__(self, n_nodes, embedding_dim, context_vector_size, hidden_size, hnet_hidden_dim = 100, hnet_n_hidden=3):
        super().__init__()

        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.context_vector_size = context_vector_size
        self.hidden_dim = hnet_hidden_dim
        self.n_hidden = hnet_n_hidden
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(num_embeddings=self.n_nodes, embedding_dim=self.embedding_dim)

        layers = [nn.Linear(self.context_vector_size + self.embedding_dim, self.hidden_dim)]

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

    def forward(self, context_vec, idx):
        emd = self.embeddings(idx)
        context_vec = context_vec.view(1, self.context_vector_size)
        hnet_vector = context_vec.expand(len(context_vec), self.embedding_dim)
        hnet_vector = torch.cat((emd, hnet_vector), dim=1)
        features = self.mlp(hnet_vector)

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

        layers = [nn.Linear(self.context_vector_size + self.embedding_dim, self.hidden_dim)]

        for _ in range(self.n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.mlp = nn.Sequential(*layers)

        self.fc1_weights = nn.Linear(self.hidden_dim, 2 * self.context_vector_size)
        self.fc1_bias = nn.Linear(self.hidden_dim, 1)

    def forward(self, context_vec, idx):
        emd = self.embeddings(idx)
        context_vec = context_vec.view(1, self.context_vector_size)
        hnet_vector = context_vec.expand(len(context_vec), self.embedding_dim).to(self.device)
        hnet_vector = torch.cat((emd, hnet_vector), dim=1)
        features = self.mlp(hnet_vector)

        weights = OrderedDict({
            "fc1.weight": self.fc1_weights(features).view(1, 2*self.context_vector_size),
            "fc1.bias": self.fc1_bias(features).view(-1),
        })

        return weights

class LR_Context(nn.Module):
    def __init__(self, input_size, context_vector_size, context_hidden_size, nn_hidden_size):
        super(LR_Context, self).__init__()

        self.input_size = input_size
        self.context_vector_size = context_vector_size
        self.context_hidden_size = context_hidden_size
        self.nn_hidden_size = nn_hidden_size

        # Context Network
        self.context_net = nn.Sequential(
                                nn.Linear(self.input_size, self.context_hidden_size),
                                nn.BatchNorm1d(self.context_hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.context_hidden_size, self.context_hidden_size),
                                nn.BatchNorm1d(self.context_hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.context_hidden_size, self.context_vector_size)
                                )

        # Neural Network
        self.fc1 = nn.Linear(self.input_size + self.context_vector_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x, context_only):
        context_vector = self.context_net(x)

        avg_context_vector = torch.mean(context_vector, dim=0)
        prediction_vector = avg_context_vector.expand(len(x), self.context_vector_size)
        prediction_vector = torch.cat((prediction_vector,x), dim=1)

        if context_only:
            return avg_context_vector

        prediction = self.fc1(prediction_vector)

        return prediction, avg_context_vector

class NN_Context(nn.Module):
    def __init__(self, input_size, context_vector_size, context_hidden_size, nn_hidden_size, dropout):
        super(NN_Context, self).__init__()

        self.input_size = input_size
        self.context_vector_size = context_vector_size
        self.context_hidden_size = context_hidden_size
        self.nn_hidden_size = nn_hidden_size
        self.dropout = dropout

        # Context Network
        self.context_net = nn.Sequential(
                                nn.Linear(self.input_size, self.context_hidden_size),
                                nn.BatchNorm1d(self.context_hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.context_hidden_size, self.context_hidden_size),
                                nn.BatchNorm1d(self.context_hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.context_hidden_size, self.context_vector_size)
                                )

        # Neural Network
        self.fc1 = nn.Linear(self.input_size + self.context_vector_size, self.nn_hidden_size)
        self.fc2 = nn.Linear(self.nn_hidden_size, self.nn_hidden_size)
        self.fc3 = nn.Linear(self.nn_hidden_size, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, x, context_only):
        context_vector = self.context_net(x)

        avg_context_vector = torch.mean(context_vector, dim=0)
        prediction_vector = avg_context_vector.expand(len(x), self.context_vector_size)
        prediction_vector = torch.cat((prediction_vector,x), dim=1)

        if context_only:
            return avg_context_vector

        prediction = self.dropout(self.relu(self.fc1(prediction_vector)))
        prediction = self.dropout(self.relu(self.fc2(prediction)))
        prediction = self.fc3(prediction)

        return prediction

class LR_plain(nn.Module):
    def __init__(self, input_size):
        super(LR_plain, self).__init__()

        self.input_size = input_size

        self.fc1 = nn.Linear(self.input_size, 1)

    def forward(self, x):
        prediction = self.fc1(x)

        return prediction

class LR(nn.Module):
    def __init__(self, input_size, bound):
        super(LR, self).__init__()

        self.bound = torch.Tensor([bound])
        self.sensitive_classes = [0, 1]
        self.input_size = input_size
        self.n_constraints = 4 # 2 * num classes
        self.dim_condition = 3 # num_classes + 1
        self.M = torch.zeros((self.n_constraints, self.dim_condition))
        self.fc1 = nn.Linear(2*self.input_size, 1)

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

class Context(nn.Module):
    def __init__(self, input_size, context_vector_size, context_hidden_size):
        super(Context, self).__init__()

        self.input_size = input_size
        self.context_vector_size = context_vector_size
        self.context_hidden_size = context_hidden_size

        self.fc1 = nn.Linear(self.input_size, self.context_hidden_size)
        self.fc2 = nn.Linear(self.context_hidden_size, self.context_hidden_size)
        self.fc3 = nn.Linear(self.context_hidden_size, self.context_vector_size)
        self.bn1 = nn.BatchNorm1d(self.context_hidden_size)
        self.bn2 = nn.BatchNorm1d(self.context_hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x, num_features):
        x_ = self.relu(self.bn1(self.fc1(x)))
        x_ = self.relu(self.bn2(self.fc2(x_)))
        context_vector = self.fc3(x_)
        avg_context_vector = torch.mean(context_vector, dim=0)

        prediction_vector = avg_context_vector.expand(len(x), num_features)
        pred_vec = torch.cat((prediction_vector, x), dim=1)


        return avg_context_vector, pred_vec

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
