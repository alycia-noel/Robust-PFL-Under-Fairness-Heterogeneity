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

        return out


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
    plt.title('ROC for NN on COMPAS')
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

cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'length_of_stay','days_b_screening_arrest', 'decile_score', 'two_year_recid']
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

d_train, d_test = train_test_split(compas, test_size=617)

data_train = TabularData(d_train[features].values, d_train[decision].values)
data_test = TabularData(d_test[features].values, d_test[decision].values)

model = NN(input_size=10)
model = model.double()

train_loader = DataLoader(data_train, shuffle = True, batch_size = 16)
test_loader = DataLoader(data_test, shuffle = False, batch_size= 16)

#optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.90, weight_decay=.001) #.0003
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=.001, last_epoch=-1)
optimizer = torch.optim.Adam(model.parameters(), lr = .0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
loss = nn.BCELoss(reduction='none') #Binary Cross Entropy loss
no_batches = len(train_loader)

loss_values =[]
test_loss_values = []
test_error, F_ERR, M_ERR = [], [], []

acc_values, test_acc, F_ACC, M_ACC = [], [], [], []

tp, tn, fp, fn, f1 = [], [], [], [], []
F_TP, F_FP, F_TN, F_FN, F_F1 = [], [], [], [], []
M_TP, M_FP, M_TN, M_FN, M_F1 = [], [], [], [], []
EOD, SPD, AOD = [], [], []

times = []

# Train model
model.train()
torch.cuda.synchronize()
epochs = 100
for epoch in range(epochs):
    start = time.time()
    running_loss = 0.0
    correct = 0.0
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
        total += data_train.y.size(0)
        correct += (preds.eq(y)).sum().item()

    accuracy = (100*correct / len(data_train))
    loss_values.append(running_loss / len(train_loader))

    torch.cuda.synchronize()
    end = time.time()
    elapsed = end - start
    times.append(elapsed)
    print('Epoch: {0}/{1};\t Loss: {2:1.3f};\tAcc:{3:1.3f};\tTime:{4:1.2f}'.format(epoch + 1, epochs, running_loss / len(train_loader), accuracy, elapsed))

    total_time = sum(times)

    # Eval Model
    model.eval()
    predictions = []
    running_loss_test = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    f_tp, f_fp, f_tn, f_fn = 0, 0, 0, 0
    m_tp, m_fp, m_tn, m_fn = 0, 0, 0, 0
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

            for i in range(len(x)):
                if x[i, 5].item() == 0:
                    if predicted_prediction[i] == 1 and labels_pred[i] == 1:
                        f_tp += 1
                        TP += 1
                    elif predicted_prediction[i] == 1 and labels_pred[i] == 0:
                        f_fp += 1
                        FP += 1
                    elif predicted_prediction[i] == 0 and labels_pred[i] == 0:
                        f_tn += 1
                        TN += 1
                    elif predicted_prediction[i] == 0 and labels_pred[i] == 1:
                        f_fn += 1
                        FN += 1
                else:
                    if predicted_prediction[i] == 1 and labels_pred[i] == 1:
                        m_tp += 1
                        TP += 1
                    elif predicted_prediction[i] == 1 and labels_pred[i] == 0:
                        m_fp += 1
                        FP += 1
                    elif predicted_prediction[i] == 0 and labels_pred[i] == 0:
                        m_tn += 1
                        TN += 1
                    elif predicted_prediction[i] == 0 and labels_pred[i] == 1:
                        m_fn += 1
                        FN += 1

    test_loss_values.append(running_loss_test / len(test_loader))
    f1_score_prediction = TP / (TP + (FP + FN) / 2)
    f1_female = f_tp / (f_tp + (f_fp + f_fn) / 2)
    f1_male = m_tp / (m_tp + (m_fp + m_fn) / 2)

    f1.append(f1_score_prediction)
    F_F1.append(f1_female)
    M_F1.append(f1_male)

    if epoch == 99:
        plt.plot(loss_values, label='Train Loss')
        plt.plot(test_loss_values, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs for NN Model on COMPAS')
        plt.legend(loc="upper right")
        plt.show()
        # if i % 354 == 0:

    #accuracy = (100 * correct) / len(data_test)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    f_acc = (f_tp + f_tn) / (f_tp + f_fp + f_fn + f_tn)
    m_acc = (m_tp + m_tn) / (m_tp + m_fp + m_fn + m_tn)

    error = (FP + FN) / (TP + FP + FN + TN)
    f_err = (f_fp + f_fn) / (f_tp + f_fp + f_fn + f_tn)
    m_err = (m_fp + m_fn) / (m_tp + m_fp + m_fn + m_tn)

    AOD.append((((f_tp / (f_tp + f_fn)) - (m_tp / (m_tp + m_fn))) + (
                (f_fp / (f_fp + f_tn)) - (m_fp / (m_fp + m_tn)))) / 2)  # average odds difference
    EOD.append((f_tp / (f_tp + f_fn)) - (m_tp / (m_tp + m_fn)))  # equal opportunity difference
    SPD.append((f_tp + f_fp) / (f_tp + f_fp + f_tn + f_fn) - (m_tp + m_fp) / (m_tp + m_fp + m_tn + m_fn))

    res = (
        pd  .DataFrame(columns = features, index = d_test.index)
            .add_suffix('_partial')
            .join(d_test)
            .assign(prediction=predictions)
            .assign(round=epoch)
    )

    results.append(res)
    test_acc.append(accuracy)
    F_ACC.append(f_acc)
    M_ACC.append(m_acc)
    test_error.append(error)
    F_ERR.append(f_err)
    M_ERR.append(m_err)

    tn.append(TN)
    tp.append(TP)
    fn.append(FN)
    fp.append(FP)

    F_TN.append(f_tn)
    F_TP.append(f_tp)
    F_FN.append(f_fn)
    F_FP.append(f_fp)

    M_TN.append(m_tn)
    M_TP.append(m_tp)
    M_FN.append(m_fn)
    M_FP.append(m_fp)

results = pd.concat(results)
average_test_acc = sum(test_acc) / len(test_acc)
female_test_acc = sum(F_ACC) / len(F_ACC)
male_test_acc = sum(M_ACC) / len(M_ACC)

print('*******************')
print('*       all       *')
print('*******************')
print('Test Accuracy: {0:1.3f}'.format(test_acc[len(test_acc) - 1]))
print('Test Error: {0:1.3f}'.format(test_error[len(test_error) - 1]))
print('TP: ', tp[len(tp)-1], 'FP: ', fp[len(fp)-1], 'TN: ', tn[len(tn)-1], 'FN: ', fn[len(fn)-1])
print('EOD: {0:1.4f}'.format(EOD[len(EOD)-1]))
print('SPD: {0:1.4f}'.format(SPD[len(SPD)-1]))
print('AOD: {0:1.4f}'.format(AOD[len(AOD)-1]))
print('F1: {0:1.3f}'.format(f1[len(f1)- 1]))
print("")
print('**********************')
print('*       Female       *')
print('**********************')
print('Test Accuracy: {0:1.3f}'.format(F_ACC[len(F_ACC) - 1]))
print('Test Error: {0:1.3f}'.format(F_ERR[len(F_ERR) - 1]))
print('TP: ', F_TP[len(F_TP)-1], 'FP: ', F_FP[len(F_FP)-1], 'TN: ', F_TN[len(F_TN)-1], 'FN: ', F_FN[len(F_FN)-1])
print('F1: {0:1.3f}'.format(F_F1[len(F_F1)- 1]))
print("")
print('********************')
print('*       Male       *')
print('********************')
print('Test Accuracy: {0:1.3f}'.format(M_ACC[len(M_ACC) - 1]))
print('Test Error: {0:1.3f}'.format(M_ERR[len(M_ERR) - 1]))
print('TP: ', M_TP[len(M_TP)-1], 'FP: ', M_FP[len(M_FP)-1], 'TN: ', M_TN[len(M_TN)-1], 'FN: ', M_FN[len(M_FN)-1])
print('F1: {0:1.3f}'.format(M_F1[len(M_F1)- 1]))

print('Train Time: {0:1.2f}'.format(total_time))

for col, encoder in encoders.items():
        results.loc[:,col] = encoder.inverse_transform(results[col])


plot_roc_curves(results, 'prediction', 'two_year_recid', size=(7, 5), fname='./results/roc.png')
