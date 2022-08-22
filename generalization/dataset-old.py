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
        min_size = 0

        if j == 0:
            min_require_size = 500
        else:
            min_require_size = 200

        # Sample according to the Dirichlet distribution
        # Minimum number of samples 1 client will have is 500 for train and 200 for test
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients - 10)]

            for k in range(2):
                idx_k = np.where(data_copy['income_class'] == k)[0]
                np.random.shuffle(idx_k)

                proportions = np.random.dirichlet(np.repeat(.1, num_clients-10))

                for i in range(10):
                    choice = np.random.choice([.1, .25, .5, 1])
                    proportions_novel = np.random.dirichlet(np.repeat(choice, 10))

                np.append(proportions, proportions_novel)

                proportions = np.array([p * (len(idx_j) < len(data_copy) / num_clients) for p, idx_j in zip(proportions, idx_batch)])

                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for i in range(num_clients):
            client_data = data_copy.iloc[idx_batch[i]]
            all_client_test_train[j].append(TabularData(client_data[features].values, client_data[labels].values))

        subsets = all_client_test_train[j]
        if j == 0:
            loader_params['shuffle'] = True
        else:
            loader_params['shuffle'] = False

        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    return dataloaders, features


