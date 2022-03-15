import time
import warnings
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix

warnings.filterwarnings("ignore")

class TabularData(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        n, m = X.shape
        self.n = n
        self.m = m
        self.X = torch.tensor(X, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class combo(nn.Module):
    def __init__(self, input_size, vector_size):
        super(combo, self).__init__()
        self.input_size = input_size
        self.vector_size = vector_size
        self.hidden_sizes = [10,10,10]
        self.hidden_size = 10
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

        return out

class HyperNet(nn.Module):
    def __init__(self, vector_size, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vector_size = vector_size

        n_hidden = 3#2#3

        layers = [
            nn.Linear(self.vector_size, hidden_dim),  # [13, 100]
        ]
        for _ in range(n_hidden):
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        #layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

        self.fc1_weights = nn.Linear(hidden_dim, 20*10)
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
    def forward(self, context_vec):
        context_vec = context_vec.view(1,self.vector_size) #[1,13]

        # Generate the weight output features by passing the context_vector through the hypernetwork mlp
        features = self.mlp(context_vec)

        weights = OrderedDict({
            "fc1.weight": self.fc1_weights(features).view(10, 20),
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

def plot_roc_curves(results, pred_col, resp_col, c, size=(7, 5), fname=None):
    plt.clf()
    plt.style.use('classic')
    plt.figure(figsize=size)

    for _, res in results.groupby('round'):
        fpr, tpr, _ = roc_curve(res[resp_col], res[pred_col])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, '-', color='orange', lw=0.5)

    fpr, tpr, _ = roc_curve(results[resp_col], results[pred_col])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, '-', color='darkorange', lw=1.5, label='ROC curve (area = %0.2f)' % roc_auc, )
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    title_roc = 'ROC for FL HN Adaptive NN on COMPAS - Client' + str(c + 1)
    plt.title(title_roc)
    plt.show()

# Import the data and visualize it (if you want using df.info())
# decile_score = risk score prediction
torch.manual_seed(0)
url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
df = pd.read_csv(url)

# Cleaning and parsing the data
# 1. If the charge date of a defendants COMPAS score was not within 30 days from when the person was arrested, we assume that because of data
#    quality reason, that we do not have the right offense
# 2. If is_recid = -1 then there was no COMPAS case found
# 3. c_charge_degree of 'O' will result in no jail time so they are removed
df_filtered = df.loc[df['days_b_screening_arrest'] <= 30]
df_filtered = df_filtered.loc[df_filtered['days_b_screening_arrest'] >= -30]
df_filtered = df_filtered.loc[df_filtered['is_recid'] != -1]
df_filtered = df_filtered.loc[df_filtered['c_charge_degree'] != "O"]
df_filtered = df_filtered.loc[df_filtered['score_text'] != 'N/A']
df_filtered['is_med_or_high_risk']  = (df_filtered['decile_score']>=5).astype(int)
df_filtered['length_of_stay'] = (pd.to_datetime(df_filtered['c_jail_out']) - pd.to_datetime(df_filtered['c_jail_in']))

cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'length_of_stay','days_b_screening_arrest', 'decile_score', 'two_year_recid']
compas = df_filtered[cols]

compas['length_of_stay'] /= np.timedelta64(1, 'D')
compas['length_of_stay'] = np.ceil(compas['length_of_stay'])

encoders = {}
for col in ['race', 'sex', 'c_charge_degree', 'score_text', 'age_cat']:
    encoders[col] = LabelEncoder().fit(compas[col])
    compas.loc[:, col] = encoders[col].transform(compas[col])

client_1 = compas[compas['age'] <= 31] #3164

client_2 = compas[compas['age'] > 31] #3008

cols_1 = client_1.columns
features_1, decision_1 = cols_1[:-1], cols_1[-1]

cols_2 = client_2.columns
features_2, decision_2 = cols_2[:-1], cols_2[-1]

clients = 2

d_train_1, d_test_1 = train_test_split(client_1, test_size=300)
data_train_1 = TabularData(d_train_1[features_1].values, d_train_1[decision_1].values)
data_test_1 = TabularData(d_test_1[features_1].values, d_test_1[decision_1].values)

d_train_2, d_test_2 = train_test_split(client_2, test_size=300)
data_train_2 = TabularData(d_train_2[features_2].values, d_train_2[decision_2].values)
data_test_2 = TabularData(d_test_2[features_2].values, d_test_2[decision_2].values)

c1_model = combo(input_size=10, vector_size=10)
c2_model = combo(input_size = 10, vector_size=10)
hnet = HyperNet(vector_size=10, hidden_dim=10)
c1_model = c1_model.double()
c2_model = c2_model.double()
hnet = hnet.double()

no_cuda=False
gpus='4'
device = torch.cuda.set_device(4)
#device = torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")

hnet=hnet.to(device)
c1_model=c1_model.to(device)
c2_model=c2_model.to(device)

optimizer = torch.optim.Adam(hnet.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) #3e-2, .01
c1_inner_optimizer = torch.optim.Adam(c1_model.parameters(), lr = .00015, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
c2_inner_optimizer = torch.optim.Adam(c2_model.parameters(), lr = .0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

loss = nn.BCELoss(reduction='mean')   #binary logarithmic loss function

c1_loss_values =[]
c1_test_loss_values = []
c1_acc_values = []
c1_test_acc =[]
c1_test_error = []
c1_tp = []
c1_tn = []
c1_fp = []
c1_fn = []
c1_times = []
c1_f1 = []
c1_results = []
c1_final_results = []
c1_all_results = []

c2_loss_values =[]
c2_test_loss_values = []
c2_acc_values = []
c2_test_acc =[]
c2_test_error = []
c2_tp = []
c2_tn = []
c2_fp = []
c2_fn = []
c2_times = []
c2_f1 = []
c2_results = []
c2_final_results = []
c2_all_results = []
# Train model
steps = 10#9#10
epochs = 25#30

print("Start")
torch.cuda.synchronize()
for step in range(steps):

    torch.save(hnet.state_dict(), 'hnet_baseline_weights.pth')

    for c in range(clients):

        if c == 0:
            print("")
            print('#######################')
            print('On Outer Iteration: ', step + 1)
            print('#######################')
            data_train = data_train_1
            data_test = data_test_1
        else:
            data_train = data_train_2
            data_test = data_test_2

        train_loader = DataLoader(data_train, shuffle=True, batch_size=16)
        test_loader = DataLoader(data_test, shuffle=False, batch_size=16)

        no_batches = len(train_loader)

        # load baseline
        hnet.load_state_dict(torch.load('hnet_baseline_weights.pth'))
        hnet.train()
        torch.cuda.synchronize()
        start_epoch = time.time()

        for epoch in range(epochs):
            if c == 0:
                model = c1_model
                inner_optimizer = c1_inner_optimizer
            if c == 1:
                model = c2_model
                inner_optimizer = c2_inner_optimizer

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0.0

            for i, (x, y) in enumerate(train_loader):
                # Generate weights on first round of the inner loop
                if epoch == 0 and i == 0:
                    context_vectors, avg_context_vector, prediction_vector = model(x.to(device), context_only=True)

                    weights = hnet(avg_context_vector)
                    net_dict = model.state_dict()
                    hnet_dict = {k: v for k, v in weights.items() if k in net_dict}
                    net_dict.update(hnet_dict)
                    model.load_state_dict(net_dict)

                    inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

                inner_optimizer.zero_grad()
                optimizer.zero_grad()

                y_ = model(x.to(device), context_only=False)
                err = loss(y_, y.unsqueeze(1).to(device))
                err = err.mean()
                running_loss += err.item() * x.size(0)
                err.backward()
                inner_optimizer.step()

                preds = y_.detach().cpu().round().reshape(1, len(y_))
                correct += (preds.eq(y)).sum().item()

            accuracy = (100 * correct / len(data_train))
            if c == 0:
                c1_acc_values.append(accuracy)
                c1_loss_values.append(running_loss / len(train_loader))
            else:
                c2_acc_values.append(accuracy)
                c2_loss_values.append(running_loss / len(train_loader))

            #print('Step: {5}/{6};\tEpoch: {0}/{1};\tLoss: {2:1.3f};\tAcc: {3:1.3f};\tPercent: {4:1.2f}%'.format(epoch+1, epochs, running_loss / len(train_loader), accuracy,(100*place)/(epochs*steps), step+1, steps))

            # Eval Model
            model.eval()
            predictions = []
            running_loss_test = 0
            TP = 0
            FP = 0
            FN = 0
            TN = 0
            total = 0.0
            correct = 0.0
            with torch.no_grad():
                for i, (x, y) in enumerate(test_loader):
                    pred = model(x.to(device), context_only=False)
                    test_err = loss(pred.flatten(), y.to(device))
                    test_err = test_err.mean()
                    running_loss_test += test_err.item() * x.size(0)

                    preds = pred.detach().cpu().round().reshape(1, len(pred))
                    predictions.extend(preds.flatten().numpy())
                    correct += (preds.eq(y)).sum().item()

                    predicted_prediction = preds.type(torch.IntTensor).numpy().reshape(-1)
                    labels_pred = y.type(torch.IntTensor).numpy().reshape(-1)

                    TP += np.count_nonzero((predicted_prediction == 1) & (labels_pred == 1))
                    FP += np.count_nonzero((predicted_prediction == 0) & (labels_pred == 1))
                    TN += np.count_nonzero((predicted_prediction == 0) & (labels_pred == 0))
                    FN += np.count_nonzero((predicted_prediction == 1) & (labels_pred == 0))

            if c == 0:
                c1_test_loss_values.append(running_loss_test / len(test_loader))
            else:
                c2_test_loss_values.append(running_loss_test / len(test_loader))

            f1_score_prediction = TP / (TP + (FP + FN) / 2)

            if c == 0:
                c1_f1.append(f1_score_prediction)
                loss_values = c1_loss_values
                test_loss_values = c1_test_loss_values
            else:
                c2_f1.append(f1_score_prediction)
                loss_values = c2_loss_values
                test_loss_values = c2_test_loss_values

            if epoch == epochs-1 and step == steps - 1:
                plt.plot(loss_values, label='Train Loss')
                plt.plot(test_loss_values, label='Test Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                title_loss = 'Loss over Epochs for FL HN Adaptive NN Model on COMPAS - Client ' + str(c + 1)
                plt.title(title_loss)
                plt.legend(loc="upper right")
                plt.show()

            accuracy = (TP+TN)/(TP+FP+FN+TN)
            error    = (FP+FN)/(TP+FP+FN+TN)
            if c == 0:
                res = (
                    pd.DataFrame(columns=features_1, index=d_test_1.index)
                        .add_suffix('_partial')
                        .join(d_test_1)
                        .assign(prediction=predictions)
                        .assign(round=epoch)
                )
            else:
                res = (
                    pd.DataFrame(columns=features_2, index=d_test_2.index)
                        .add_suffix('_partial')
                        .join(d_test_2)
                        .assign(prediction=predictions)
                        .assign(round=epoch)
                )

            if c == 0:
                c1_results.append(res)
                if epoch == epochs - 1:
                    c1_final_results.append(res)
                c1_test_acc.append(accuracy)
                c1_test_error.append(error)
                c1_tn.append(TN)
                c1_tp.append(TP)
                c1_fn.append(FN)
                c1_fp.append(FP)
            else:
                c2_results.append(res)
                if epoch == epochs - 1:
                    c2_final_results.append(res)
                c2_test_acc.append(accuracy)
                c2_test_error.append(error)
                c2_tn.append(TN)
                c2_tp.append(TP)
                c2_fn.append(FN)
                c2_fp.append(FP)

        if c == 0:
            c1_all_results = pd.concat(c1_results)
            c1_final_final_results = pd.concat(c1_final_results)
            average_test_acc = sum(c1_test_acc) / len(c1_test_acc)
            print('Client: ', c + 1)
            print('Test Accuracy: {0:1.3f};'.format(c1_test_acc[len(c1_test_acc) - 1]))
            print('Test Error: {0:1.3f};'.format(c1_test_error[len(c1_test_error) - 1]))
            print('TP: ', c1_tp[len(c1_tp) - 1], 'FP: ', c1_fp[len(c1_fp) - 1], 'TN: ', c1_tn[len(c1_tn) - 1], 'FN: ',
                  c1_fn[len(c1_fn) - 1])
            print('F1: {0:1.3f}'.format(c1_f1[len(c1_f1) - 1]))
            print("")
            for col, encoder in encoders.items():
                c1_all_results.loc[:, col] = encoder.inverse_transform(c1_all_results[col])

            torch.cuda.synchronize()
            end_epoch = time.time()
            elapsed = end_epoch - start_epoch
            c1_times.append(elapsed)
        else:
            c2_all_results = pd.concat(c2_results)
            c2_final_final_results = pd.concat(c2_final_results)
            average_test_acc = sum(c2_test_acc) / len(c2_test_acc)
            print('Client: ', c + 1)
            print('Test Accuracy: {0:1.3f};'.format(c2_test_acc[len(c2_test_acc) - 1]))
            print('Test Error: {0:1.3f};'.format(c2_test_error[len(c2_test_error) - 1]))
            print('TP: ', c2_tp[len(c2_tp) - 1], 'FP: ', c2_fp[len(c2_fp) - 1], 'TN: ', c2_tn[len(c2_tn) - 1], 'FN: ',
                  c2_fn[len(c2_fn) - 1])
            print('F1: {0:1.3f}'.format(c2_f1[len(c2_f1) - 1]))
            for col, encoder in encoders.items():
                c2_all_results.loc[:, col] = encoder.inverse_transform(c2_all_results[col])

            torch.cuda.synchronize()
            end_epoch = time.time()
            elapsed = end_epoch - start_epoch
            c2_times.append(elapsed)

        optimizer.zero_grad()
        final_state = model.state_dict()

        # calculating delta theta
        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

        # calculating phi gradient
        hnet_grads = torch.autograd.grad(
            list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
        )

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g

        # torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

print("Time Client 1: ", sum(c1_times), "Time Client 2: ", sum(c2_times))

plot_roc_curves(c1_final_final_results, 'prediction', 'two_year_recid', c=0, size=(7, 5), fname='./results/roc.png')
plot_roc_curves(c2_final_final_results, 'prediction', 'two_year_recid', c=1, size=(7, 5), fname='./results/roc.png')
