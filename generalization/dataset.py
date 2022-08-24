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

        alpha = [x+1 for x in range(30)]

        #for _, a in enumerate(alpha):
        np.random.seed(0)
        label_distribution_first_half = np.random.dirichlet([5,1], 90).tolist() #(2, 90)
        #label_distribution_second_half = np.random.dirichlet([1,5], 45).tolist()

        label_distribution = []
        label_distribution = label_distribution_first_half

        # for i in range(45):
        #     label_distribution.append(label_distribution_first_half[i])
        #     label_distribution.append(label_distribution_second_half[i])

        zero_distribution = [z[0] for _,z in enumerate(label_distribution)]
        one_distribution = [z[1] for _,z in enumerate(label_distribution)]

        proportions_novel = np.random.dirichlet([r, r], 10)
        zero_distribution = np.append(zero_distribution, proportions_novel[:,0])
        one_distribution = np.append(one_distribution, proportions_novel[:,1])

        # normalize so sum_i p_i,j = 1 for all classes j
        zero_distribution = [float(i)/sum(zero_distribution) for i in zero_distribution]
        one_distribution = [float(i)/sum(one_distribution) for i in one_distribution]


        novel_clients = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

        #for i, client in enumerate(novel_clients):
        P = []
        Q = []

        for i, num in enumerate(novel_clients):
            P.append([zero_distribution[num], one_distribution[num]])

        for i in range(90):
            Q.append([zero_distribution[i], one_distribution[i]])

        #print('Alpha:', a, 'TV:', TotalVariation(Q, P, zero_distribution, one_distribution))

        total_variation.append(TotalVariation(Q, P, zero_distribution, one_distribution))


        class_idcs = [np.argwhere(np.array(data_copy['income_class'] == y)).flatten() for y in range(num_classes)]

        proportions = [zero_distribution, one_distribution]
        client_idcs = [[] for _ in range(num_clients)]

        for c, fracs in zip(class_idcs, proportions):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

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


