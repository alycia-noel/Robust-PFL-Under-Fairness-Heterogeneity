import copy

import fairlearn.datasets
import os
import torch
import warnings
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from collections import OrderedDict
from torch.utils.data import Dataset
from experiments.new.node import BaseNodes
from sklearn.datasets import make_classification
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class LR(nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.input_size=input_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 1),
            nn.Sigmoid()
        )

    def forward(self, X, **kwargs):
        return self.model(X)


class SampleWeightLR(NeuralNetClassifier):
    def __init__(self, *args, criterion__reduce=False, **kwargs):
        super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

    def fit(self, X, y, sample_weight=None):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy().astype("float32")
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if sample_weight is not None and isinstance(
                sample_weight, (pd.DataFrame, pd.Series)
        ):
            sample_weight = sample_weight.to_numpy()
        y = y.reshape([-1, 1])

        sample_weight = (
            sample_weight if sample_weight is not None else np.ones_like(y)
        )
        X = {"X": X, "sample_weight": sample_weight}
        return super().fit(X, y)

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy().astype("float32")
        return(super().predict_proba(X) > 0.5).astype(float)

    def get_loss(self, y_pred, y_true, X, *args, **kwargs):
        loss_unreduced = super().get_loss(
            y_pred, y_true.float(), X, *args, **kwargs
        )
        sample_weight = X["sample_weight"]
        sample_weight = sample_weight.to(loss_unreduced.device).unsqueeze(-1)
        loss_reduced = (sample_weight * loss_unreduced).mean()
        return loss_reduced

    def get_params(self, deep=True, **kwargs):
        params = []

        for name, param in self.module_.named_parameters():
            params.append(param.cpu().detach().numpy().flatten())

        return params


def plot_data(Xs, Ys):
    labels = np.unique(Xs["example_sensitive_feature"])

    for l in labels:
        label_string = str(l.item())
        mask = Xs["example_sensitive_feature"] == l
        plt.scatter(Xs[mask].credit_score_feature, Ys[mask], label=str("Label=" + label_string))
        plt.xlabel("Credit Score")
        plt.ylabel("Got Loan")

    plt.legend()
    plt.show()

device = "cuda:4" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed(0)

CURRENT_DIR = os.path.abspath(os.path.dirname(__name__))
TRAIN_DATA_FILE = os.path.join(CURRENT_DIR, 'adult.data')
TEST_DATA_FILE = os.path.join(CURRENT_DIR, 'adult.test')

def get_data(path):
    data_types = OrderedDict([
        ("age", "int"),
        ("workclass", "category"),
        ("final_weight", "int"),
        ("education", "category"),
        ("education_num", "int"),
        ("marital_status","category"),
        ("occupation", "category"),
        ("relationship", "category"),
        ("race" ,"category"),
        ("sex", "category"),
        ("capital_gain", "float"),
        ("capital_loss", "int"),
        ("hours_per_week", "int"),
        ("native_country", "category"),
        ("income_class", "category"),
    ])

    data = pd.read_csv(
                path,
                names=data_types,
                index_col=None,
                dtype=data_types,
                comment='|',
                skipinitialspace=True,
                na_values={
                    'capital_gain':99999,
                    'workclass':'?',
                    'native_country':'?',
                    'occupation':'?',
                },
            )

    data['income_class'] = data.income_class.str.rstrip('.').astype('category')

    data = data.drop('final_weight', axis=1)

    data = data.drop_duplicates()

    data = data.dropna(how='any', axis=0)

    data.capital_gain = data.capital_gain.astype(int)

    data.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], ['not married', 'married', 'married', 'married', 'not married', 'not married', 'not married'], inplace = True)
    encoders = {}
    for col in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income_class']:
        encoders[col] = LabelEncoder().fit(data[col])
        data.loc[:, col] = encoders[col].transform(data[col])

    return data

# To get parameters: net.coef_, to get bias: net.intercept_[:,None]

train_data = get_data(TRAIN_DATA_FILE)
test_data  = get_data(TEST_DATA_FILE)

cols = train_data.columns
features, label = cols[:-1], cols[-1]
X, Y = train_data[features].to_numpy(dtype='f'), train_data[label].to_numpy(dtype='f')
X_test, Y_test = test_data[features].to_numpy(dtype='f'), test_data[label].to_numpy(dtype='f')

net = SampleWeightLR(LR(X.shape[1]), max_epochs=5, optimizer=optim.Adam, lr=.001, batch_size=32, train_split=None, iterator_train__shuffle=True, criterion=nn.BCELoss, device=device)
moment=DemographicParity()

verification_moment = copy.deepcopy(moment)
unmitigated = copy.deepcopy(net)
# unmitigated.fit(X, Y)

num_predictors = 2

first_sweep=GridSearch(net, constraints=moment, grid_size=num_predictors)
first_sweep.fit(X, Y, sensitive_features=X[:,8])

lambda_vecs = first_sweep.lambda_vecs_
print(lambda_vecs[0])
actual_multipliers = [lambda_vecs[col][("+", "all", 1)]-lambda_vecs[col][("-", "all", 1)] for col in lambda_vecs]
print(actual_multipliers)

lambda_best = first_sweep.lambda_vecs_[first_sweep.best_idx_][("+", "all", 1)] - first_sweep.lambda_vecs_[first_sweep.best_idx_][("-", "all", 1)]
print("lambda_best =", lambda_best)
print("coefficients =", first_sweep.predictors_[first_sweep.best_idx_].get_params())
Y_first_predict = first_sweep.predict(X_test)

second_sweep_multipliers = np.linspace(lambda_best - 0.5, lambda_best + 0.5, 10) #31

iterables = [['+', '-'], ['all'], [0, 1]]
midx = pd.MultiIndex.from_product(iterables, names=['sign', 'event', 'group_id'])

second_sweep_lambdas = []
for l in second_sweep_multipliers:
    nxt = pd.Series(np.zeros(4), index=midx)
    if l < 0:
        nxt[("-", "all", 1)] = abs(l)
    else:
        nxt[("+", "all", 1)] = l
    second_sweep_lambdas.append(nxt)

multiplier_df = pd.concat(second_sweep_lambdas, axis=1)
second_sweep=GridSearch(net, constraints=DemographicParity(), grid_size=num_predictors, grid=multiplier_df)

second_sweep.fit(X, Y, sensitive_features=X[:,8])
lambda_best_second = second_sweep.lambda_vecs_[second_sweep.best_idx_][("+", "all", 1)] \
                     -second_sweep.lambda_vecs_[second_sweep.best_idx_][("-", "all", 1)]
print("lambda_best =", lambda_best_second)
print("coefficients =", second_sweep.predictors_[second_sweep.best_idx_].get_params())
Y_second_predict = second_sweep.predict(X_test)
print(Y_first_predict)
print(Y_second_predict)
# assert len(first_sweep.predictors_) == num_predictors
#
# verification_moment.load_data(X_test, Y_test, sensitive_features=X_test[:,8])
# gamma_unmitigated = verification_moment.gamma(lambda x: unmitigated.predict(x))
# gamma_mitigated = verification_moment.gamma(lambda x: first_sweep.predict(x))
#
# for idx in gamma_mitigated.index:
#     assert abs(gamma_mitigated[idx]) <= abs(
#         gamma_unmitigated[idx]
#     ), "Checking {0}".format(idx)






