from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from experiments.utils import get_device
# This class is the architecture for the hypernetwork
# embedding_dim is default to 12
# n_hidden is default to 3

# Don't need to change much here, just the sizes of the layers and the number of them, also the in_channels and so on
class Hyper(nn.Module):
    def __init__(
            self, embedding_dim, in_channels=3, out_dim=10, hidden_dim=100, n_hidden=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # create the number of needed layers, so initial linear, activation functions (ReLU), and the needed hidden
        # linear layers
        layers = [
             nn.Linear(embedding_dim, hidden_dim), #[13, 100]
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim), #[100, 100]
            )

        # create the entire mlp from the above layers
        self.mlp = nn.Sequential(*layers) #final layer is [100,100]

        # Generating a way to save the weights and biases by creating linear layers of the right size so that they
        # can be passed to the clients to load into their network
        self.fc1_weights = nn.Linear(hidden_dim, ((32 * 32 * 3 + 13) * 100)) #[100, 308500]
        self.fc1_bias = nn.Linear(hidden_dim, hidden_dim) #[100, 100]
        self.fc2_weights = nn.Linear(hidden_dim, (hidden_dim*hidden_dim)) #[100, 10000]
        self.fc2_bias = nn.Linear(hidden_dim, hidden_dim) #[100, 100]
        self.out_weights = nn.Linear(hidden_dim, 10*hidden_dim) #[100, 10]
        self.out_bias = nn.Linear(hidden_dim, 10) #[110, 10] - this isn't right, need to fix

    # Do a forward pass
    def forward(self, context_vec):
        context_vec = context_vec.view(1, 13) #[1,13]

        # Generate the weight output features by passing the context_vector through the hypernetwork mlp
        features = self.mlp(context_vec) #[1, 100]

        weights = OrderedDict({
            "fc1.weight": self.fc1_weights(features).view(100, 32 * 32 * 3 + 13),
            "fc1.bias": self.fc1_bias(features).view(-1),
            "fc2.weight": self.fc2_weights(features).view(100, 100),
            "fc2.bias": self.fc2_bias(features).view(-1),
            "out.weight": self.out_weights(features).view(10, 100),
            "out.bias": self.out_bias(features).view(-1),
        })

        return weights

# Create the architecture that each client uses
class TargetAndContext(nn.Module):
    def __init__(self, n_hidden_nodes, keep_rate = 0, input_size = 3072, hidden_size = 200, vector_size = 13):  #in_channels=3, n_kernels=16, out_dim=10):
        super(TargetAndContext, self).__init__()

        self.n_hidden_nodes = n_hidden_nodes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vector_size = vector_size

        if not keep_rate:
            keep_rate = 0.5
        self.keep_rate = keep_rate

        # Context network
        self.context_fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.context_relu1 = nn.LeakyReLU()
        self.context_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_relu2 = nn.LeakyReLU()
        self.context_context = nn.Linear(self.hidden_size, self.vector_size)

        # Target Network
        self.fc1 = nn.Linear(input_size + vector_size, self.n_hidden_nodes) #[3104, 100], bias is [100]
        self.fc1_drop = nn.Dropout(1 - self.keep_rate)
        self.fc2 = nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes) #[100, 100], bias is [100]
        self.fc2_drop = nn.Dropout(1 - self.keep_rate)
        self.out = nn.Linear(self.n_hidden_nodes, 10) #[100, 10]

    def forward(self, x, contextonly):
        # pass through context vector
        x = torch.flatten(x, 1)  # flatten for processing [64 x 3072]
        hidden1 = self.context_fc1(x)
        relu1 = self.context_relu1(hidden1)
        hidden2 = self.context_fc2(relu1)
        relu2 = self.context_relu2(hidden2)
        context_vector = self.context_context(relu2)

        ###### adaptive prediction
        avg_context_vector = torch.mean(context_vector, dim=0)  # [13]

        prediction_vector = avg_context_vector.expand(len(x), self.vector_size)
        prediction_vector = torch.cat((prediction_vector, x), dim=1)  # shape = [64, 3104]

        # If we just need the context vector for the hypernet
        if contextonly == True:
            return context_vector, avg_context_vector, prediction_vector

        x = nn.functional.relu(self.fc1(prediction_vector))
        x = self.fc1_drop(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return nn.functional.log_softmax(self.out(x))


# The context network, generates a vector that is passed to the hypernetwork
# class ContextNetwork(nn.Module):
#     def __init__(self, input_size = 3072, hidden_size= 200, vector_size = 13):
#         super(ContextNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu1 = nn.LeakyReLU()
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.relu2 = nn.LeakyReLU()
#         self.context = nn.Linear(hidden_size, vector_size)
#
#
#         self.vector_size = vector_size
#
#     def forward(self,x):
#         x = torch.flatten(x, 1) # flatten for processing [64 x 3072]
#         hidden1 = self.fc1(x)
#         relu1 = self.relu1(hidden1)
#         hidden2 = self.fc2(relu1)
#         relu2 = self.relu2(hidden2)
#         context_vector = self.context(relu2)
#
#         ###### adaptive prediction
#         avg_context_vector = torch.mean(context_vector, dim=0) #[13]
#
#         prediction_vector = avg_context_vector.expand(len(x), self.vector_size)
#         prediction_vector = torch.cat((prediction_vector, x), dim=1) # shape = [64, 3104]
#
#         return context_vector, avg_context_vector, prediction_vector

