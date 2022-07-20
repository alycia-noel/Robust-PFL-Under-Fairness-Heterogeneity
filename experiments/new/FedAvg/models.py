from collections import OrderedDict
import torch
from torch import nn
import numpy as np

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
                        self.M[i, j + 1] = 1.0
                        self.M[i, -1] = -1
                    else:
                        self.M[i, j] = -1
                        self.M[i, -1] = 1

        elif self.fairness == 'eo':
            self.n_constraints = self.num_classes * self.num_classes * 2
            self.dim_condition = self.num_classes * (self.num_classes + 1)
            self.M = torch.zeros((self.n_constraints, self.dim_condition))

            self.c = torch.zeros(self.n_constraints)
            element_k_a = self.sensitive_classes + [None]

            for i_a, a_0 in enumerate(self.sensitive_classes):
                for i_y, y_0 in enumerate(self.y_classes):
                    for i_s, s in enumerate([-1, 1]):
                        for j_y, y_1 in enumerate(self.y_classes):
                            for j_a, a_1 in enumerate(element_k_a):
                                i = i_a * (2 * self.num_classes) + i_y * 2 + i_s
                                j = j_y + self.num_classes * j_a
                                self.M[i, j] = self.__element_M(a_0, a_1, y_1, y_1, s)

    def __element_M(self, a0, a1, y0, y1, s):
        if a0 is None or a1 is None:
            x = y0 == y1
            return -1 * s * x
        else:
            x = (a0 == a1) & (y0 == y1)
            return s * float(x)

    def mu_f(self, out, sensitive, y):
        expected_values_list = []

        if self.fairness == 'eo':
            for u in self.sensitive_classes:
                for v in self.y_classes:
                    idx_true = (y == v) * (sensitive == u)
                    if torch.sum(idx_true.type(torch.FloatTensor)) == 0:
                        expected_values_list.append(out.mean() * 0)
                    else:
                        expected_values_list.append(out[idx_true].mean())

            for v in self.y_classes:
                idx_true = y == v
                if torch.sum(idx_true.type(torch.FloatTensor)) == 0:
                    expected_values_list.append(out.mean() * 0)
                else:
                    expected_values_list.append(out[idx_true].mean())


        elif self.fairness == 'dp':
            sensitive = sensitive.view(out.shape)
            compare = torch.tensor([]).to(sensitive.device)
            for v in self.sensitive_classes:
                idx_true = sensitive == v
                if torch.equal(out[idx_true], compare):
                    expected_values_list.append(out.mean() * 0)
                else:
                    expected_values_list.append(out[idx_true].mean())
            expected_values_list.append(out.mean())

        return torch.stack(expected_values_list)

    def M_mu_q(self, pred, sensitive, y):
        return torch.mv(self.M.to(pred.device), self.mu_f(pred, sensitive, y)) - self.c.to(pred.device)

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

