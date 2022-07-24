from collections import OrderedDict
import torch
import sys
from torch import nn
import numpy as np
import torch.nn.functional as F

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
            #print("out: ", out)
            compare = torch.tensor([]).to(sensitive.device)

            for u in self.sensitive_classes:
                for v in self.y_classes:
                    idx_true = (y == v) * (sensitive == u)
                    #print('out[idx_true]: ', out[idx_true])
                    #print('mean: ', out[idx_true].mean())
                    #print('mean * 0: ', out.mean() * 0)
                    if torch.equal(out[idx_true], compare):
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
                    expected_values_list.append(out.mean()*0)
                else:
                    expected_values_list.append(out[idx_true].mean())
            expected_values_list.append(out.mean())

        return torch.stack(expected_values_list)

    def M_mu_q(self, pred, sensitive, y):
        const = torch.mv(self.M.to(pred.device), self.mu_f(pred, sensitive, y)) - self.c.to(pred.device)
        return const

    def forward(self, x, s, y):
        prediction = torch.sigmoid(self.fc1(x))

        if self.fairness == 'none':
            m_mu_q = None
        else:
            m_mu_q = self.M_mu_q(prediction, s, y)

        return prediction, m_mu_q

# class Constraint(torch.nn.Module):
#     def __init__(self, fair, bound):
#         super().__init__()
#
#         self.bound = 1 / bound
#         self.fair = fair
#
#         if self.fair == 'dp':
#             self.register_parameter(name='lmbda', param=torch.nn.Parameter(torch.rand((4,1))))
#         elif self.fair == 'eo':
#             self.register_parameter(name='lmbda', param=torch.nn.Parameter(torch.rand((8,1))))
#
#     def forward(self, value):
#         loss = torch.matmul(self.lmbda.T, value)
#
#         return loss

class Constraint(torch.nn.Module):
    def __init__(self, bound, relation, name=None, multiplier_act=F.relu, alpha=0., start_val=0., size=4):
        """
        Adds a constraint to a loss function by turning the loss into a lagrangian.
        Alpha is used for a moving average as described in [1].
        Note that this is similar as using an optimizer with momentum.
        [1] Rezende, Danilo Jimenez, and Fabio Viola.
            "Taming vaes." arXiv preprint arXiv:1810.00597 (2018).
        Args:
            bound: Constraint bound.
            relation (str): relation of constraint,
                using naming convention from operator module (eq, le, ge).
                Defaults to 'ge'.
            name (str, optional): Constraint name
            multiplier_act (optional): When using inequality relations,
                an activation function is used to force the multiplier to be positive.
                I've experimented with ReLU, abs and softplus, softplus seems the most stable.
                Defaults to F.ReLU.
            alpha (float, optional): alpha of moving average, as in [1].
                If alpha=0, no moving average is used.
            start_val (float, optional): Start value of multiplier. If an activation function
                is used the true start value might be different, because this is pre-activation.
        """
        super().__init__()
        self.name = name
        self.size = size

        if isinstance(bound, (int, float)):
            self.bound = torch.Tensor([bound])
        elif isinstance(bound, list):
            self.bound = torch.Tensor(bound)
        else:
            self.bound = bound

        if relation in {'ge', 'le', 'eq'}:
            self.relation = relation
        else:
            raise ValueError('Unknown relation: {}'.format(relation))

        if self.relation == 'eq' and multiplier_act is not None:
            sys.stderr.write(
                "WARNING using an activation that maps to R+ with an equality \
                 constraint turns it into an inequality constraint"
            )

        self.lmbda = torch.nn.Parameter(torch.rand(self.size,1))
        self._act = multiplier_act

        self.alpha = alpha
        self.avg_value = None

    @property
    def multiplier(self):
        if self._act is not None:
            return self._act(self.lmbda)
        return self.lmbda

    def forward(self, value):
        loss = value - self.bound.to(value.device)
        return self.multiplier.T * loss

class Wrapper:
    """
    Simple class wrapper around  obj = obj_type(*args, **kwargs).
    Overwrites methods from obj with methods defined in Wrapper,
    else uses method from obj.
    """

    def __init__(self, obj_type, *args, **kwargs):
        self.obj = obj_type(*args, **kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.obj, attr)

    def __repr__(self):
        return 'Wrapped(' + self.obj.__repr__() + ')'

class ConstraintOptimizer(Wrapper):
    """
    Pytorch Optimizers only do gradient descent, but lagrangian needs
    gradient ascent for the multipliers. ConstraintOptimizer changes
    step() method of optimizer to do ascent instead of descent.
    I've gotten the best results using RMSprop with lr
    around 1e-3 and Constraint alpha=0.5.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *args, **kwargs):
        super().__init__(optimizer, *args, **kwargs)

    def step(self, *args, **kwargs):
        # Maximize instead of minimize.
        for group in self.obj.param_groups:
            for p in group['params']:
                p.grad = -p.grad
        self.obj.step(*args, **kwargs)

    def __repr__(self):
        return 'ConstraintOptimizer (' + self.obj.__repr__() + ')'