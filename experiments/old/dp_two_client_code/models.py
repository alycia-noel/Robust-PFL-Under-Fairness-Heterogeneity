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

class LR_context(nn.Module):
    def __init__(self, input_size, vector_size):
        super(LR_context, self).__init__()
        self.input_size = input_size
        self.vector_size = vector_size
        self.hidden_size = 100

        # Logistic Regression
        self.fc1 = nn.Linear(self.input_size + self.vector_size, 1)

        # Context Network
        self.context_fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.context_relu1 = nn.LeakyReLU()
        self.context_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_relu2 = nn.LeakyReLU()
        self.context_fc3 = nn.Linear(self.hidden_size, self.vector_size)

    def forward(self, x):
        # Pass through context network
        hidden1 = self.context_fc1(x)
        relu1 = self.context_relu1(hidden1)
        hidden2 = self.context_fc2(relu1)
        relu2 = self.context_relu2(hidden2)
        context_vector = self.context_fc3(relu2)

        ###### adaptive prediction
        avg_context_vector = torch.mean(context_vector, dim=0)
        prediction_vector = avg_context_vector.expand(len(x), self.vector_size)
        prediction_vector = torch.cat((prediction_vector, x), dim=1)

        x1 = self.fc1(prediction_vector)
        y = torch.sigmoid(x1)
        return y, x1

class LR_combo(nn.Module):
    def __init__(self, input_size, vector_size, hidden_size):
        super(LR_combo, self).__init__()
        self.input_size = input_size
        self.vector_size = vector_size
        self.hidden_size = hidden_size # 100

        # Logistic Regression
        self.fc1 = nn.Linear(self.input_size + self.vector_size, 1)

        # Context Network
        self.context_fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.context_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_fc3 = nn.Linear(self.hidden_size, self.vector_size)
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)
        self.context_relu = nn.LeakyReLU()

    def forward(self, x, context_only):
        # Pass through context network
        hidden1 = self.context_fc1(x)
        norm1 = self.batchnorm(hidden1)
        relu1 = self.context_relu(norm1)
        hidden2 = self.context_fc2(relu1)
        norm2 = self.batchnorm(hidden2)
        relu2 = self.context_relu(norm2)
        context_vector = self.context_fc3(relu2)

        ###### adaptive prediction
        avg_context_vector = torch.mean(context_vector, dim=0)
        prediction_vector = avg_context_vector.expand(len(x), self.vector_size)
        prediction_vector = torch.cat((prediction_vector, x), dim=1)

        if context_only:
            return context_vector, avg_context_vector, prediction_vector

        x1 = self.fc1(prediction_vector)
        y = torch.sigmoid(x1)

        return y, x1

class LR_HyperNet(nn.Module):
    def __init__(self, vector_size, hidden_dim, num_hidden):
        super().__init__()

        self.embedding_dim = 9
        self.embeddings = nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_dim)

        self.hidden_dim = hidden_dim
        self.vector_size = vector_size

        n_hidden = num_hidden

        layers = [
            nn.Linear(self.vector_size + self.embedding_dim, hidden_dim),  # [13, 100]
        ]
        # layers = [
        #     nn.Linear(self.embedding_dim, hidden_dim),  # [13, 100]
        # ]
        for _ in range(n_hidden):
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )

        self.mlp = nn.Sequential(*layers)

        self.fc1_weights = nn.Linear(hidden_dim, (2*self.vector_size)) #[input size, 1]
        self.fc1_bias = nn.Linear(hidden_dim, 1)


    # Do a forward pass
    def forward(self, context_vec, idx):
        emd = self.embeddings(idx)

        context_vec = context_vec.view(1, self.vector_size)  # [1,13]
        hnet_vector = context_vec.expand(len(context_vec), self.embedding_dim)
        hnet_vector = torch.cat((emd, hnet_vector), dim=1)

        # Generate the weight output features by passing the context_vector through the hypernetwork mlp
        features = self.mlp(hnet_vector)

        weights = OrderedDict({
            "fc1.weight": self.fc1_weights(features).view(1,2*self.vector_size),
            "fc1.bias": self.fc1_bias(features).view(-1),
        })

        return weights

class NN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64,10,10], dropout_rate = .45):
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

class NN_context(nn.Module):
    def __init__(self, input_size,vector_size, hidden_sizes=[10,10,10]):
        super(NN_context, self).__init__()
        self.input_size = input_size
        self.vector_size = vector_size
        self.dropout_rate = .45
        self.hidden_size = 9

        # neural network
        self.fc1 = nn.Linear(self.input_size+self.vector_size, hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc3 = nn.Linear(hidden_sizes[2], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[2])
        self.fc5 = nn.Linear(hidden_sizes[2], 1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_rate)

        # Context Network
        self.context_fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.context_relu1 = nn.LeakyReLU()
        self.context_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_relu2 = nn.LeakyReLU()
        self.context_fc3 = nn.Linear(self.hidden_size, self.vector_size)

    def forward(self, x):
        # Pass through context network
        hidden1 = self.context_fc1(x)
        relu1 = self.context_relu1(hidden1)
        hidden2 = self.context_fc2(relu1)
        relu2 = self.context_relu2(hidden2)
        context_vector = self.context_fc3(relu2)

        ###### adaptive prediction
        avg_context_vector = torch.mean(context_vector, dim=0)
        prediction_vector = avg_context_vector.expand(len(x), self.vector_size)
        prediction_vector = torch.cat((prediction_vector, x), dim=1)

        x1 = F.relu(self.fc1(prediction_vector))
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

class NN_combo(nn.Module):
    def __init__(self, input_size, vector_size, hidden_size):
        super(NN_combo, self).__init__()
        self.input_size = input_size
        self.vector_size = vector_size
        self.hidden_sizes = [10,10,10]
        self.hidden_size = hidden_size
        self.dropout_rate = .45

        # NN
        self.fc1 = nn.Linear(self.input_size + self.vector_size, self.hidden_sizes[1])
        self.fc2 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.fc3 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[2])
        self.fc4 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[2])
        self.fc5 = nn.Linear(self.hidden_sizes[2], 1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.dropout_rate)

        # Context Network
        self.context_fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.context_relu1 = nn.LeakyReLU()
        self.context_fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_relu2 = nn.LeakyReLU()
        self.context_fc3 = nn.Linear(self.hidden_size, self.vector_size)

    def forward(self, x, context_only):
        # Pass through context network
        hidden1 = self.context_fc1(x)
        relu1 = self.context_relu1(hidden1)
        hidden2 = self.context_fc2(relu1)
        relu2 = self.context_relu2(hidden2)
        context_vector = self.context_fc3(relu2)

        ###### adaptive prediction
        avg_context_vector = torch.mean(context_vector, dim=0)
        prediction_vector = avg_context_vector.expand(len(x), self.vector_size)
        prediction_vector = torch.cat((prediction_vector, x), dim=1)

        if context_only:
            return context_vector, avg_context_vector, prediction_vector

        x1 = F.relu(self.fc1(prediction_vector))
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

class NN_HyperNet(nn.Module):
    def __init__(self, vector_size, hidden_dim,num_hidden):
        super().__init__()
        self.embedding_dim = 9
        self.embeddings = nn.Embedding(num_embeddings=2, embedding_dim=self.embedding_dim)

        self.hidden_dim = hidden_dim
        self.vector_size = vector_size

        n_hidden = num_hidden

        layers = [
            nn.Linear(self.vector_size + self.embedding_dim, hidden_dim),  # [13, 100]
        ]
        # layers = [
        #     nn.Linear(self.embedding_dim, hidden_dim),  # [13, 100]
        # ]
        for _ in range(n_hidden):
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

        self.fc1_weights = nn.Linear(hidden_dim, 18*10)
        self.fc1_bias = nn.Linear(hidden_dim, 10)
        self.fc2_weights = nn.Linear(hidden_dim, 10*10)
        self.fc2_bias = nn.Linear(hidden_dim, 10)
        self.fc3_weights = nn.Linear(hidden_dim, 10*10)
        self.fc3_bias = nn.Linear(hidden_dim, 10)
        self.fc4_weights = nn.Linear(hidden_dim, 10*10)
        self.fc4_bias = nn.Linear(hidden_dim, 10)
        self.fc5_weights = nn.Linear(hidden_dim, 10)
        self.fc5_bias = nn.Linear(hidden_dim, 1)


    # Do a forward pass
    def forward(self, context_vec, idx):
        emd = self.embeddings(idx)

        context_vec = context_vec.view(1, self.vector_size)  # [1,13]
        hnet_vector = context_vec.expand(len(context_vec), self.embedding_dim)
        hnet_vector = torch.cat((emd, hnet_vector), dim=1)

        # Generate the weight output features by passing the context_vector through the hypernetwork mlp
        features = self.mlp(hnet_vector)

        weights = OrderedDict({
            "fc1.weight": self.fc1_weights(features).view(10, 18),
            "fc1.bias": self.fc1_bias(features).view(-1),
            "fc2.weight": self.fc2_weights(features).view(10, 10),
            "fc2.bias": self.fc2_bias(features).view(-1),
            "fc3.weight": self.fc3_weights(features).view(10, 10),
            "fc3.bias": self.fc3_bias(features).view(-1),
            "fc4.weight": self.fc4_weights(features).view(10, 10),
            "fc4.bias": self.fc4_bias(features).view(-1),
            "fc5.weight": self.fc5_weights(features).view(1, 10),
            "fc5.bias": self.fc5_bias(features).view(-1)
        })

        return weights