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

class LR(nn.Module):
    def __init__(self, input_size, vector_size):
        super(LR, self).__init__()
        self.input_size = input_size
        self.vector_size = vector_size
        self.hidden_size = 10

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
        return y


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
    title_roc = 'ROC for Adaptive LR on COMPAS - Client '+ str(c + 1)
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

model = LR(input_size=10, vector_size=10)
model = model.double()

#optimizer = torch.optim.SGD(model.parameters(), lr=3.e-4, momentum=.5, weight_decay=3.e-5) #best: 3.e-4, .5, 3.e-5,   3e-3 is the original lr, momentum = .5 and .4 (more like what we want, still kind of jagged), .9 (too high, really wack loss), wd=5e-5
optimizer = torch.optim.Adam(model.parameters(), lr = 5.e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
print("Start")
for c in range(clients):

    if c == 0:
        data_train = data_train_1
        data_test = data_test_1
    else:
        data_train = data_train_2
        data_test = data_test_2

    train_loader = DataLoader(data_train, shuffle=True, batch_size=32)
    test_loader = DataLoader(data_test, shuffle=False, batch_size=32)

    loss = nn.BCELoss(reduction='mean')   #binary logarithmic loss function
    no_batches = len(train_loader)
    loss_values =[]
    results = []
    test_loss_values = []
    acc_values = []
    test_acc =[]
    test_error = []
    tp = []
    tn = []
    f1 = []
    fp = []
    fn = []
    times = []

    # Train model

    model.train() #warm-up
    torch.cuda.synchronize()
    for epoch in range(100): #original 250, best:700
        start_epoch = time.time()
        running_loss = 0.0
        correct = 0
        total = 0.0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_ = model(x)
            err = loss(y_.flatten(), y)
            err = err.mean()
            running_loss += err.item() * x.size(0)
            err.backward()
            optimizer.step()

            preds = y_.round().reshape(1, len(y_))
            correct += (preds.eq(y)).sum().item()

        accuracy = (100 * correct / len(data_train))
        acc_values.append(accuracy)
        loss_values.append(running_loss / len(train_loader))

        torch.cuda.synchronize()
        end_epoch = time.time()
        elapsed = end_epoch - start_epoch
        times.append(elapsed)
        #print('Epoch: {0}/{1};\t Loss: {2:1.3f};\tAcc:{3:1.3f};\tTime:{4:1.2f}'.format(epoch + 1, 100, running_loss / len(train_loader), accuracy, elapsed))

        total_time = sum(times)

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
                pred = model(x)
                test_err = loss(pred.flatten(), y)
                test_err = test_err.mean()
                running_loss_test += test_err.item() * x.size(0)

                preds = pred.round().reshape(1, len(pred))
                predictions.extend(preds.flatten().numpy())
                correct += (preds.eq(y)).sum().item()

                predicted_prediction = preds.type(torch.IntTensor).numpy().reshape(-1)
                labels_pred = y.type(torch.IntTensor).numpy().reshape(-1)

                TP += np.count_nonzero((predicted_prediction == 1) & (labels_pred == 1))
                FP += np.count_nonzero((predicted_prediction == 0) & (labels_pred == 1))
                TN += np.count_nonzero((predicted_prediction == 0) & (labels_pred == 0))
                FN += np.count_nonzero((predicted_prediction == 1) & (labels_pred == 0))

        test_loss_values.append(running_loss_test / len(test_loader))

        f1_score_prediction = TP / (TP + (FP + FN) / 2)
        f1.append(f1_score_prediction)

        if epoch == 99:
            plt.plot(loss_values, label='Train Loss')
            plt.plot(test_loss_values, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            title_loss = 'Loss over Epochs for Adaptive LR Model on COMPAS - Client ' + str(c + 1)
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

        results.append(res)
        test_acc.append(accuracy)
        test_error.append(error)
        tn.append(TN)
        tp.append(TP)
        fn.append(FN)
        fp.append(FP)
    results = pd.concat(results)
    average_test_acc = sum(test_acc) / len(test_acc)
    print('Client: ', c + 1)
    print('Test Accuracy: {0:1.3f};'.format(test_acc[len(test_acc) - 1]))
    print('Test Error: {0:1.3f};'.format(test_error[len(test_error) - 1]))
    print('TP: ', tp[len(tp)-1], 'FP: ', fp[len(fp)-1], 'TN: ', tn[len(tn)-1], 'FN: ', fn[len(fn)-1])
    print('F1: {0:1.3f}'.format(f1[len(f1)- 1]))
    print('Train Time: {0:1.2f}'.format(total_time))

    for col, encoder in encoders.items():
            results.loc[:,col] = encoder.inverse_transform(results[col])

    plot_roc_curves(results, 'prediction', 'two_year_recid', c,  size=(7, 5), fname='./results/roc.png')
