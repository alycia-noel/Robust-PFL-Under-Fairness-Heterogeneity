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

    return data

def clean_and_encode_dataset(data, data_name):

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

    return data

def get_dataset(data_name):

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
    #train_data = train_data.sort_values('workclass').reset_index(drop=True)
    #test_data = test_data.sort_values('workclass').reset_index(drop=True)
    #splits = [train_data.index[np.searchsorted(train_data['workclass'], .5, side='right')],test_data.index[np.searchsorted(test_data['workclass'], .5, side='right')+1]]
    datasets = [train_data, test_data]

    cols = train_data.columns
    features, labels = cols[:-1], cols[-1]

    return datasets, features, labels

def gen_random_loaders(data_name, num_clients, bz):
    loader_params = {"batch_size": bz, "shuffle": True, "pin_memory": True, "num_workers": 0}

    dataloaders = []

    datasets, features, labels = get_dataset(data_name)
    all_client_test_train = [[], []]

    for j, data_copy in enumerate(datasets):

        if j == 0:
            min = 4000
        else:
            min = 1400

        amounts = [random.randint(min, int((len(data_copy)/2))) for i in range(num_clients)]

        for i in range(num_clients):
            client_data = data_copy.sample(n = amounts[i], replace=True, ignore_index = False)

            all_client_test_train[j].append(TabularData(client_data[features].values, client_data[labels].values))

        subsets = all_client_test_train[j]
        if j == 0:
            loader_params['shuffle'] = True
        else:
            loader_params['shuffle'] = False

        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    return dataloaders, features


