import os
import random
from torch.utils.data import Dataset
import torch.utils.data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import numpy as np

class TabularData(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        n, m = X.shape
        self.n = n
        self.m = m
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]



def read_dataset(path, data_types, data_name):
    if data_name == 'adult':
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
    elif data_name == 'compas':
        url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
        data = pd.read_csv(url)
    return data

def clean_and_encode_dataset(data, data_name):
    if data_name == 'adult':
        data['income_class'] = data.income_class.str.rstrip('.').astype('category')

        data = data.drop('final_weight', axis=1)

        data = data.drop_duplicates()

        data = data.dropna(how='any', axis=0)

        data.capital_gain = data.capital_gain.astype(int)

        data.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married',
                      'Separated', 'Widowed'],
                     ['not married', 'married', 'married', 'married', 'not married', 'not married', 'not married'],
                     inplace=True)
        data.replace(['Federal-gov', 'Local-gov', 'State-gov'], ['government', 'government', 'government'],
                     inplace=True)
        data.replace(['Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'Without-pay'],
                     ['private', 'private', 'private', 'private', 'private'], inplace=True)
        encoders = {}

        for col in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income_class']:
            encoders[col] = LabelEncoder().fit(data[col])
            data.loc[:, col] = encoders[col].transform(data[col])

    elif data_name == 'compas':
        data = data.loc[data['days_b_screening_arrest'] <= 30]
        data = data.loc[data['days_b_screening_arrest'] >= -30]
        data = data.loc[data['is_recid'] != -1]
        data = data.loc[data['c_charge_degree'] != "O"]
        data = data.loc[data['score_text'] != 'N/A']
        data['is_med_or_high_risk'] = (data['decile_score'] >= 5).astype(int)
        data['length_of_stay'] = (
                pd.to_datetime(data['c_jail_out']) - pd.to_datetime(data['c_jail_in']))

        cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'length_of_stay', 'days_b_screening_arrest', 'decile_score', 'two_year_recid']

        data = data[cols]

        data['length_of_stay'] /= np.timedelta64(1, 'D')
        data['length_of_stay'] = np.ceil(data['length_of_stay'])

        encoders = {}

        for col in ['sex','race', 'c_charge_degree', 'score_text', 'age_cat']:
            encoders[col] = LabelEncoder().fit(data[col])
            data.loc[:, col] = encoders[col].transform(data[col])

    return data

def get_dataset(data_name, num_clients):
    if data_name == 'adult':

        CURRENT_DIR = os.path.abspath(os.path.dirname(__name__))
        TRAIN_DATA_FILE = os.path.join(CURRENT_DIR, 'adult.data')
        TEST_DATA_FILE = os.path.join(CURRENT_DIR, 'adult.test')

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

        train_data = clean_and_encode_dataset(read_dataset(TRAIN_DATA_FILE, data_types, 'adult'), 'adult')  #(32561, 14)
        test_data = clean_and_encode_dataset(read_dataset(TEST_DATA_FILE, data_types, 'adult'), 'adult')    #(16281, 14)
        train_data = train_data.sort_values('workclass').reset_index(drop=True)
        test_data = test_data.sort_values('workclass').reset_index(drop=True)
        splits = [train_data.index[np.searchsorted(train_data['workclass'], 1)],test_data.index[np.searchsorted(test_data['workclass'], 1)]]
        datasets = [train_data, test_data]

        cols = train_data.columns
        features, labels = cols[:-1], cols[-1]

        return datasets, splits, features, labels

    elif data_name == 'compas':
        data = clean_and_encode_dataset(read_dataset(None, None, 'compas'), 'compas')
        train_data, test_data = train_test_split(data, test_size=.15, train_size=.85, random_state=42, shuffle=True)
        train_data = train_data.sort_values('age').reset_index(drop=True)
        test_data = test_data.sort_values('age').reset_index(drop=True)
        splits = [train_data.index[np.searchsorted(train_data['age'], 31, side='left')],
                  test_data.index[np.searchsorted(test_data['age'], 31, side='left')]]

        datasets = [train_data, test_data]

        cols = train_data.columns
        features, labels = cols[:-1], cols[-1]

    return datasets, splits, features, labels

def gen_random_loaders(data_name, num_clients, bz):
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}

    dataloaders = []

    datasets, splits, features, labels = get_dataset(data_name, num_clients)

    all_client_test_train = [[], []]

    first_set = random.randint(1,num_clients-1)
    second_set = abs(num_clients - first_set)

    assert first_set > 0
    assert second_set > 0

    for j, data_copy in enumerate(datasets):
        amount_two = int((data_copy.shape[0] - splits[j]) / second_set)

        for i in range(num_clients):
            if i < first_set:
                amount = int((splits[j] - 1) / first_set)
            else:
                amount = amount_two

            # last client gets the remainders
            if i == num_clients - 1:
                client_data = data_copy
            else:
                client_data = data_copy[0:amount]


            all_client_test_train[j].append(TabularData(client_data[features].values, client_data[labels].values))
            data_copy = data_copy[amount:]

        subsets = all_client_test_train[j]
        if j == 0:
            loader_params['shuffle'] = True
        else:
            loader_params['shuffle'] = False
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    return dataloaders, features


