from torch.utils.data import Dataset
from collections import namedtuple
import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_list_like


class ColumnAlreadyDroppedWarning(UserWarning):
    """Warning used if a column is attempted to be dropped twice."""
from scipy.stats import multivariate_normal
import torch, random, copy, os

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LogReg(torch.nn.Module):
    """
    Logistic regression model.
    """
    def __init__(self, num_features, num_classes, seed = 123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x.float())
        probas = torch.sigmoid(logits)
        return probas.type(torch.FloatTensor), logits

class Mlp(torch.nn.Module):
    """
    MLP model.
    """
    def __init__(self, num_features, num_classes, seed = 123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.linear1 = torch.nn.Linear(num_features, 4)
        self.linear2 = torch.nn.Linear(4, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.linear1(x.float())
        out = self.relu(out)
        out = self.linear2(out)
        probs = torch.sigmoid(out)
        return probs.type(torch.FloatTensor), out


DATA_HOME_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'raw')
COMPAS_URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'

def fetch_compas(data_home=None, binary_race=False,
                 usecols=['sex', 'age', 'age_cat', 'race', 'juv_fel_count',
                          'juv_misd_count', 'juv_other_count', 'priors_count',
                          'c_charge_degree', 'c_charge_desc'],
                 dropcols=[], numeric_only=False, dropna=True):
    cache_path = os.path.join(data_home or DATA_HOME_DEFAULT,
                              os.path.basename(COMPAS_URL))
    if os.path.isfile(cache_path):
        df = pd.read_csv(cache_path, index_col='id')
    else:
        df = pd.read_csv(COMPAS_URL, index_col='id')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)

    # Perform the same preprocessing as the original analysis:
    # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    df = df[(df.days_b_screening_arrest <= 30)
          & (df.days_b_screening_arrest >= -30)
          & (df.is_recid != -1)
          & (df.c_charge_degree != 'O')
          & (df.score_text != 'N/A')]

    for col in ['sex', 'age_cat', 'race', 'c_charge_degree', 'c_charge_desc']:
        df[col] = df[col].astype('category')

    # 'Survived' < 'Recidivated'
    cats = ['Survived', 'Recidivated']
    df.two_year_recid = df.two_year_recid.replace([0, 1], cats).astype('category')
    df.two_year_recid = df.two_year_recid.cat.set_categories(cats, ordered=True)

    if binary_race:
        # 'African-American' < 'Caucasian'
        df.race = df.race.cat.set_categories(['African-American', 'Caucasian'],
                                             ordered=True)

    # 'Male' < 'Female'
    df.sex = df.sex.astype('category').cat.reorder_categories(
            ['Male', 'Female'], ordered=True)

    return standardize_dataset(df, prot_attr=['sex', 'race'],
                               target='two_year_recid', usecols=usecols,
                               dropcols=dropcols, numeric_only=numeric_only,
                               dropna=dropna)

def check_already_dropped(labels, dropped_cols, name, dropped_by='numeric_only',
                          warn=True):
    if not is_list_like(labels):
        labels = [labels]
    str_labels = [c for c in labels if isinstance(c, str)]
    already_dropped = dropped_cols.intersection(str_labels)
    if warn and any(already_dropped):
        warnings.warn("Some column labels from `{}` were already dropped by "
                "`{}`:\n{}".format(name, dropped_by, already_dropped.tolist()),
                ColumnAlreadyDroppedWarning, stacklevel=2)
    return [c for c in labels if not isinstance(c, str) or c not in already_dropped]

def standardize_dataset(df, prot_attr, target, sample_weight=None, usecols=[],
                       dropcols=[], numeric_only=False, dropna=True):
    orig_cols = df.columns
    if numeric_only:
        for col in df.select_dtypes('category'):
            if df[col].cat.ordered:
                df[col] = df[col].factorize(sort=True)[0]
                df[col] = df[col].replace(-1, np.nan)
        df = df.select_dtypes(['number', 'bool'])
    nonnumeric = orig_cols.difference(df.columns)

    prot_attr = check_already_dropped(prot_attr, nonnumeric, 'prot_attr')
    if len(prot_attr) == 0:
        raise ValueError("At least one protected attribute must be present.")
    df = df.set_index(prot_attr, drop=False, append=True)

    target = check_already_dropped(target, nonnumeric, 'target')
    if len(target) == 0:
        raise ValueError("At least one target must be present.")
    y = pd.concat([df.pop(t) for t in target], axis=1).squeeze()  # maybe Series

    # Column-wise drops
    orig_cols = df.columns
    if usecols:
        usecols = check_already_dropped(usecols, nonnumeric, 'usecols')
        df = df[usecols]
    unused = orig_cols.difference(df.columns)

    dropcols = check_already_dropped(dropcols, nonnumeric, 'dropcols', warn=False)
    dropcols = check_already_dropped(dropcols, unused, 'dropcols', 'usecols', False)
    df = df.drop(columns=dropcols)

    # Index-wise drops
    if dropna:
        notna = df.notna().all(axis=1) & y.notna()
        df = df.loc[notna]
        y = y.loc[notna]

    if sample_weight is not None:
        return namedtuple('WeightedDataset', ['X', 'y', 'sample_weight'])(
                          df, y, df.pop(sample_weight).rename('sample_weight'))
    print(namedtuple('Dataset', ['X', 'y'])(df, y))
    return namedtuple('Dataset', ['X', 'y'])(df, y)

fetch_compas()
