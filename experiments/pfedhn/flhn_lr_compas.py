import time
import warnings
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
        self.hidden_size = 11

        # Logistic Regression
        self.fc1 = nn.Linear(self.input_size + self.vector_size, 1)

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

        x1 = self.fc1(prediction_vector)
        y = torch.sigmoid(x1)
        return y

class HyperNet(nn.Module):
    def __init__(self, vector_size, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.vector_size = vector_size

        n_hidden = 2

        layers = [
            nn.Linear(self.vector_size, hidden_dim),  # [13, 100]
        ]
        for _ in range(n_hidden):
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

        self.fc1_weights = nn.Linear(hidden_dim, (2*self.vector_size)) #[input size, 1]
        self.fc1_bias = nn.Linear(hidden_dim, 1)


    # Do a forward pass
    def forward(self, context_vec):
        context_vec = context_vec.view(1,self.vector_size) #[1,13]

        # Generate the weight output features by passing the context_vector through the hypernetwork mlp
        features = self.mlp(context_vec)

        weights = OrderedDict({
            "fc1.weight": self.fc1_weights(features).view(1,2*self.vector_size),
            "fc1.bias": self.fc1_bias(features).view(-1),
        })

        return weights

def plot_roc_curves(results, pred_col, resp_col, size=(7, 5), fname=None):
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
    plt.title('ROC for FL HN Adaptive LR on COMPAS')
    #if fname is not None:
    #    plt.savefig(fname)
    #else:
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

cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'length_of_stay','days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid']
compas = df_filtered[cols]

compas['length_of_stay'] /= np.timedelta64(1, 'D')
compas['length_of_stay'] = np.ceil(compas['length_of_stay'])

cols = compas.columns
features, decision = cols[:-1], cols[-1]

encoders = {}
for col in ['race', 'sex', 'c_charge_degree', 'score_text', 'age_cat']:
    encoders[col] = LabelEncoder().fit(compas[col])
    compas.loc[:, col] = encoders[col].transform(compas[col])

results = []
final_results = []
all_results = []
d_train, d_test = train_test_split(compas, test_size=617)
data_train = TabularData(d_train[features].values, d_train[decision].values)
data_test = TabularData(d_test[features].values, d_test[decision].values)

model = combo(input_size=11, vector_size=11)
hnet = HyperNet(vector_size=11, hidden_dim=11) #100
model = model.double()
hnet = hnet.double()

no_cuda=False
gpus='0'
device = torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")

hnet=hnet.to(device)
model=model.to(device)
#2e-2
optimizer = torch.optim.Adam(hnet.parameters(), lr=3e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
inner_optimizer = torch.optim.SGD(model.parameters(), lr=3.e-4, momentum=.5, weight_decay=3.e-5) #best: 3.e-4, .5, 3.e-5,   3e-3 is the original lr, momentum = .5 and .4 (more like what we want, still kind of jagged), .9 (too high, really wack loss), wd=5e-5

loss = nn.BCELoss(reduction='mean')   #binary logarithmic loss function

train_loader = DataLoader(data_train, shuffle = True, batch_size = 32)
test_loader = DataLoader(data_test, shuffle = False, batch_size= 32)

no_batches = len(train_loader)
loss_values =[]
test_loss_values = []
acc_values = []
test_acc =[]
test_error = []
tp = []
tn = []
fp = []
fn = []
times = []

# Train model
steps = 10
epochs = 70
place = 0
torch.cuda.synchronize()
for step in range(steps):

    print('#######################')
    print('On Outer Iteration: ', step+1)
    print('#######################')
    hnet.train()
    torch.cuda.synchronize()
    start_epoch = time.time()

    for epoch in range(epochs):
        place = place + 1
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

            classes = torch.argmax(y_, dim=1)

            for j in classes:
                if float(classes[j].item()) == y[j].item():
                    correct = correct + 1

        accuracy = (100 * correct / len(data_train))
        acc_values.append(accuracy)
        loss_values.append(running_loss / len(train_loader))

        print('Step: {5}/{6};\tEpoch: {0}/{1};\tLoss: {2:1.3f};\tAcc: {3:1.3f};\tPercent: {4:1.2f}%'.format(epoch+1, epochs, running_loss / len(train_loader), accuracy,(100*place)/(epochs*steps), step+1, steps))

        # Eval Model
        model.eval()
        predictions = []
        running_loss_test = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                pred = model(x.to(device), context_only=False)
                predictions.extend(pred.detach().cpu().flatten().numpy())
                test_err = loss(pred.flatten(), y.to(device))
                test_err = test_err.mean()
                running_loss_test += test_err.item() * x.size(0)

        test_loss_values.append(running_loss_test / len(test_loader))

        if epoch == epochs - 1:
            plt.plot(loss_values, label='Train Loss')
            plt.plot(test_loss_values, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss over Epochs for FL HN Adaptive LR Model on COMPAS')
            plt.legend(loc="upper right")
            plt.show()

        my_rounded_list = [round(elem) for elem in predictions]
        CM = confusion_matrix(data_test.y, my_rounded_list)

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]

        accuracy = (TP+TN)/(TP+FP+FN+TN)
        error    = (FP+FN)/(TP+FP+FN+TN)
        res = (
        pd  .DataFrame(columns = features, index = d_test.index)
            .add_suffix('_partial')
            .join(d_test)
            .assign(prediction=predictions)
            .assign(round=(step+1)*epoch)
        )

        results.append(res)
        if epoch == epochs - 1:
            final_results.append(res)
        test_acc.append(accuracy)
        test_error.append(error)
        tn.append(TN)
        tp.append(TP)
        fn.append(FN)
        fp.append(FP)

    all_results = pd.concat(results)
    average_test_acc = sum(test_acc) / len(test_acc)
    print('Test Accuracy: ',test_acc[0], test_acc[len(test_acc) - 1], average_test_acc)
    print('Test Error: ', test_error[0], test_error[len(test_error) - 1])
    print('TP: ', tp[len(tp)-1], 'TN: ', tn[len(tn)-1], 'FP: ', fp[len(fp)-1], 'FN: ', fn[len(fn)-1])
    for col, encoder in encoders.items():
            all_results.loc[:,col] = encoder.inverse_transform(all_results[col])

    torch.cuda.synchronize()
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    times.append(elapsed)

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

print(sum(times))
plot_roc_curves(all_results, 'prediction', 'two_year_recid', size=(7, 5), fname='./results/roc.png')
