import os
import random
from torch.utils.data import Dataset
import torch.utils.data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import OrderedDict
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

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

def TotalVariation(training, novel, z, o):

    training_tree = KDTree(training)

    all_nn_indicies = training_tree.query(novel, k=1, eps=.01, p=2)[1].tolist()
    all_nns = [training[idx] for idx in all_nn_indicies]

    abs_difference_zero = [abs(n[0] - t[0]) for n, t in zip(novel, all_nns)]
    abs_difference_one = [abs(n[0] - t[0]) for n, t in zip(novel, all_nns)]

    tv = [(abs_z + abs_o)/2 for abs_z, abs_o in zip(abs_difference_zero, abs_difference_one)]

    return(np.average(tv))

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

def gen_random_loaders(data_name, num_clients, bz, r):
    loader_params = {"batch_size": bz, "shuffle": True, "pin_memory": True, "num_workers": 0}

    dataloaders = []
    total_variation = []

    datasets, features, labels = get_dataset(data_name)
    all_client_test_train = [[], []]

    for j, data_copy in enumerate(datasets):
        num_classes = 2

        #alpha = [x for x in [.1, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]]

        #for _, a in enumerate(alpha):
        zero_distribution_train = np.random.beta(.5, .5, 90) #(2, 90)

        one_distribution_train = [1-x for x in zero_distribution_train]

        #label_distribution = [[x,y] for x,y in zip(zero_distribution, one_distribution)]

        zero_distribution_novel = np.random.beta(r, r, 10)
        one_distribution_novel = [1-x for x in zero_distribution_novel]

        zero_distribution = np.append(zero_distribution_train, zero_distribution_novel)
        one_distribution = np.append(one_distribution_train, one_distribution_novel)

        # normalize so sum_i p_i,j = 1 for all classes j
        # zero_distribution = [float(i)/sum(zero_distribution) for i in zero_distribution]
        # one_distribution = [float(i)/sum(one_distribution) for i in one_distribution]

        novel_clients = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

        #for i, client in enumerate(novel_clients):
        P = []
        Q = []

        for i, num in enumerate(novel_clients):
            P.append([zero_distribution[num], one_distribution[num]])

        for i in range(90):
            Q.append([zero_distribution[i], one_distribution[i]])

        #print('Alpha:', a, 'TV:', TotalVariation(Q, P, zero_distribution, one_distribution))
        #exit(1)
        total_variation.append(TotalVariation(Q, P, zero_distribution, one_distribution))

        zero_class_idcs = np.argwhere(np.array(data_copy['income_class'] == 0)).flatten().tolist()
        one_class_idcs = np.argwhere(np.array(data_copy['income_class'] == 1)).flatten().tolist()
        client_idcs = [[] for _ in range(num_clients)]

        for i in range(num_clients):
            num_data_points = np.random.choice(range(100, 4000))
            num_zero_points = int(zero_distribution[i] * num_data_points)
            num_one_points = int(one_distribution[i] * num_data_points)

            zero_idcs = random.choices(zero_class_idcs, k=num_zero_points)
            one_idcs = random.choices(one_class_idcs, k=num_one_points)

            client_idcs[i] = np.append(zero_idcs, one_idcs)

        for i in range(num_clients):
            client_data = data_copy.iloc[client_idcs[i]]
            all_client_test_train[j].append(TabularData(client_data[features].values, client_data[labels].values))

        subsets = all_client_test_train[j]
        if j == 0:
            loader_params['shuffle'] = True
        else:
            loader_params['shuffle'] = False

        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))
    return dataloaders, features, total_variation


