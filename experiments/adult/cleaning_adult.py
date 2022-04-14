import os
from torch.utils.data import Dataset
import torch.utils.data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import OrderedDict

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


def read_dataset(path, data_types):
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

def clean_and_encode_dataset(data):
    data['income_class'] = data.income_class.str.rstrip('.').astype('category')

    data = data.drop('final_weight', axis=1)

    data = data.drop_duplicates()

    data = data.dropna(how='any', axis=0)

    data.capital_gain = data.capital_gain.astype(int)

    encoders = {}
    for col in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income_class']:
        encoders[col] = LabelEncoder().fit(data[col])
        data.loc[:, col] = encoders[col].transform(data[col])

    return data

def get_dataset(val_size):
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

    train = clean_and_encode_dataset(read_dataset(TRAIN_DATA_FILE,data_types))  #(32561, 14)
    test = clean_and_encode_dataset(read_dataset(TEST_DATA_FILE, data_types))    #(16281, 14)

    train_size = len(train) - val_size
    train_set, val_set = torch.utils.data.random_split(train, [train_size, val_size])


get_dataset(100)