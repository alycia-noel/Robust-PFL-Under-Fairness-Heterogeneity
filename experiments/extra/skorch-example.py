import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from torch import nn
import torch.nn.functional as F
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate
from fairlearn.metrics import MetricFrame, selection_rate, count
import pandas as pd
from sklearn import metrics as skm
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#make data
X, y = make_classification(1000, 20, n_informative=10, random_state=0, n_classes=2)
X = X.astype(np.float32)
A = X[:, 4]
#X = np.delete(X, 4, axis = 1)
y = y.astype(np.int64)

# Work around indexing bug
# X_train = X_train.reset_index(drop=True)
# A_train = A_train.reset_index(drop=True)
# X_test = X_test.reset_index(drop=True)
# A_test = A_test.reset_index(drop=True)

#set grid search params
params = {
    'lr':[0.01, 0.02],
    'max_epochs':[10,20],
    'module__num_units': [10,20],
}


def plot_data(Xs, Ys, A):
    labels = np.unique(A)

    for l in labels:
        label_string = str(l.item())
        mask = A == l
        plt.scatter(Xs[mask, :], Ys[mask], label=str("Label=" + label_string))
        plt.xlabel("Credit Score")
        plt.ylabel("Got Loan")

    plt.legend()
    plt.show()

class MyModule(nn.Module):
    def __init__(self, num_units=10, nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(20, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(num_units, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        #X = self.output(X).flatten()
        X = F.softmax(self.output(X), dim=1) #required for NLL
        #X = torch.sigmoid(X)
        return X

auc = EpochScoring(scoring='roc_auc', lower_is_better=False) #this is after each epoch
f1 = EpochScoring(scoring='f1', lower_is_better=False)

net = NeuralNetClassifier(MyModule, criterion=nn.NLLLoss, max_epochs=20, lr=0.02, iterator_train__shuffle=True, device='cuda:4', callbacks=[auc, f1])

#gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')


net.fit(X, y)
Y_predict_unfair = net.predict(X)
plot_data(X, Y_predict_unfair, A)


first_sweep=GridSearch(net,
                       constraints=DemographicParity(),
                       grid_size=7)

first_sweep.fit(X, y, sensitive_features=A)

lambda_vecs = first_sweep.lambda_vecs_
actual_multipliers = [lambda_vecs[col][("+", "all", 2)]-lambda_vecs[col][("-", "all", 2)] for col in lambda_vecs]

first_sweep_sensitive_feature_weights = [
            predictor.coef_[0][1] for predictor in first_sweep.predictors_]
plt.scatter(actual_multipliers, first_sweep_sensitive_feature_weights)
plt.xlabel("Lagrange Multiplier")
plt.ylabel("Weight of Protected Attribute in Model")
plt.show()

Y_first_predict = first_sweep.predict(X)
plot_data(X, Y_first_predict, A)

#print(gs.best_score_, gs.best_params_)
#y_proba = net.predict_proba(X)

#to save
#net.save_params(f_params=filename)
#new_net.load_params(file_name)