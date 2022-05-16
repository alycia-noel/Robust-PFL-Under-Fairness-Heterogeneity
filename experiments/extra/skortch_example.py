import copy
import os
import torch
import warnings
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import OrderedDict
from skorch import NeuralNetClassifier
from fairlearn.reductions import DemographicParity, EqualizedOdds
from experiments.new.fairlearn_modified.exponentiated_gradient import ExponentiatedGradient
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import MetricFrame, mean_prediction, selection_rate, demographic_parity_difference, demographic_parity_ratio, true_negative_rate, true_positive_rate, false_negative_rate, false_positive_rate, equalized_odds_difference
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

class LR(nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.input_size=input_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, 1),
            nn.Sigmoid()
        )

        # self.context = nn.Sequential(
        #     nn.Linear(self.input_size, 25),
        #     nn.BatchNorm1d(25),
        #     nn.ReLU(),
        #     nn.Linear(25, 25),
        #     nn.BatchNorm1d(25),
        #     nn.ReLU(),
        #     nn.Linear(25, self.input_size)
        # )

    def forward(self, X, **kwargs):
        # this is where we can call the context vector
        # i.e., get context, concat with x, then call self.model(x)
        # context_vector = self.context(X)
        # avg_context_vector = torch.mean(context_vector, dim=0)
        # prediction_vector = avg_context_vector.expand(len(X), len(X))
        # pred_vec = torch.cat((prediction_vector, X), dim=1)

        return self.model(X)


class SampleWeightLR(NeuralNetClassifier):
    def __init__(self, *args, criterion__reduce=False, **kwargs):
        super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)
        # can we make a self.constraints = {}  and for each model fit store the context vectors and when we recall the best
        # model also get the best constraint vectors so we can agg and send to the hypernework?

    #need to alter fit loop here to get collection of context vectors and we can update the hnet here as well
    # need run single epoch

    def fit(self, X, y, sample_weight=None):
        #
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

        return super().fit(X, y) #this super().fit refers to the skorch fit method. This fit method is called during the fit (best_h) of expograd

    def predict(self, X):
        # we can totally override this to be whatever we want!
        # i.e., just use predict function from trainer code
        # just need to check the other end for what is being returned (i.e., vector, matrix, dataframe ...)
        #here we need to get updated network param from HNET using context
        # 1. get context
        # 2. get param from hnet
        # 3. instantiate params
        # 4. classify

        # will most likely just need to copy the predict_proba class from sklearn and modify
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

    # def get_params(self, deep=True, **kwargs):
    #     params = []
    #
    #     for name, param in self.module_.named_parameters():
    #         params.append(param.cpu().detach().numpy().flatten())
    #
    #     return params


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

def get_metrics_df(models_dict, y_true, group):
    metrics_dict = {
        "Overall selection rate": (lambda x: selection_rate(y_true, x)),
        "Demographic parity difference": (lambda x: demographic_parity_difference(y_true, x, sensitive_features=group)),
        "Demographic parity ratio": (lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group)),
        "------": (lambda x: ""),
        "Overall balanced error rate": (lambda x: 1-balanced_accuracy_score(y_true, x)),
        "Balanced error rate difference": (lambda x: MetricFrame(metrics=balanced_accuracy_score, y_true=y_true, y_pred=x, sensitive_features=group).difference(method='between_groups')),
        " ------": (lambda x: ""),
        "Equalized odds difference": ( lambda x: equalized_odds_difference(y_true, x, sensitive_features=group)),
        "  ------": (lambda x: "")
    }
    df_dict = {}
    for metric_name, metric_func in metrics_dict.items():
        df_dict[metric_name] = [metric_func(preds) for model_name, preds in models_dict.items()]
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())

def summary_as_df(name, summary):
    a = summary.by_group
    a['overall'] = summary.overall
    return pd.DataFrame({name: a})

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
    data.replace(['Federal-gov', 'Local-gov', 'State-gov'], ['government', 'government', 'government'], inplace = True)
    data.replace(['Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'Without-pay'], ['private', 'private', 'private', 'private', 'private'], inplace=True)
    encoders = {}

    for col in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income_class']:
        encoders[col] = LabelEncoder().fit(data[col])
        data.loc[:, col] = encoders[col].transform(data[col])

    return data

mf = []
roc = []
metrics = []
auc_sel = []

train_data = get_data(TRAIN_DATA_FILE)
test_data  = get_data(TEST_DATA_FILE)

num_clients = 4
first_set = 2
second_set = 2

train_data = train_data.sort_values('workclass').reset_index(drop=True)
split = train_data.index[np.searchsorted(train_data['workclass'], 1)]
data_copy_train = train_data
data_copy_test = test_data
datasets = [data_copy_train, data_copy_test]
all_client_test_train, all_client_train, all_client_test = [[],[]], [], []

for j, data_copy in enumerate(datasets):
    amount_two = int((data_copy.shape[0] - split) / second_set)

    for i in range(num_clients):
        if i < first_set:
            amount = int((split-1) / first_set)
        else:
            amount = amount_two

        if i == num_clients - 1:
            client_data = data_copy
        else:
            client_data = data_copy[0:amount]

        all_client_test_train[j].append(client_data)
        data_copy = data_copy[amount:]


cols = train_data.columns
features_xa, features_x, label = cols[:-1], cols[:-2], cols[-1]

all_client_train, all_client_test = all_client_test_train

for j in range(num_clients):
    # can we go ahead and get the context vector for the full dataset her? then just pass x + context in the fit?
    # would we benefit from the learning process? - No, I don't think so. We would only be able to update the context network at
    # the end of training the main network when it should be updated iteratively alongside it.
        # get batch
        # get context
        # classify
        # loss
        # backprop
        # repeat
    # can we just copy the code for exponentiated gradient and modify it? I think this may be the best bet rather than trying to update
    # sklearn fit
    print("Training client:", j+1)
    net = SampleWeightLR(LR(len(features_xa)), max_epochs=25 , optimizer=optim.Adam, lr=.001, batch_size=32, train_split=None, iterator_train__shuffle=True, criterion=nn.BCELoss, device=device)
    expgrad_xa = ExponentiatedGradient(net, constraints=DemographicParity(difference_bound=.01), eps=0.05, nu=1e-6)

    # this calls the fit function of exponentiated gradient class
    expgrad_xa.fit(all_client_train[j][features_xa], all_client_train[j][label], sensitive_features=all_client_train[j]['sex'])

    # this calls the predict function of the exponentatied gradient class
    scores_expgrad_xa = pd.Series(expgrad_xa.predict(all_client_test[j][features_xa]), name="scores_expgrad_xa")

    # need to figure out how to use context here
    metric_frame_auc = MetricFrame(metrics=roc_auc_score, y_true=all_client_test[j][label], y_pred=scores_expgrad_xa, sensitive_features=all_client_test[j]['sex'])
    metric_frame_mean = MetricFrame(metrics=mean_prediction, y_true=all_client_test[j][label], y_pred=scores_expgrad_xa, sensitive_features=all_client_test[j]['sex'])
    auc = summary_as_df("auc", metric_frame_auc)
    sel = summary_as_df("selection", metric_frame_mean)

    auc.loc['disparity'] = '-'
    sel.loc['disparity'] = (sel.loc[1] - sel.loc[0]).abs()

    mf.append( MetricFrame({
        'TPR': true_positive_rate,
        'FPR': false_positive_rate,
        'TNR': true_negative_rate,
        'FNR': false_negative_rate},
        all_client_test[j][label], scores_expgrad_xa, sensitive_features=all_client_test[j]['sex']))

    models_dict = {'expgrad': (scores_expgrad_xa)}
    roc.append(roc_auc_score(all_client_train[j][label], expgrad_xa.predict(all_client_train[j][features_xa])))
    metrics.append(get_metrics_df(models_dict, all_client_test[j][label], all_client_test[j]['sex']))
    auc_sel.append(pd.concat([auc, sel], axis=1))

for i in range(num_clients):
    print("Results for client:", i+1)
    print('-'*19)
    print(mf[i].by_group)
    print(roc[i])
    print(metrics[i])
    print(auc_sel[i])


