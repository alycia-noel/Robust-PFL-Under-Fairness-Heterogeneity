import os
import random
from torch.utils.data import Dataset
import torch.utils.data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import SubsetRandomSampler

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

        data.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], ['not married', 'married', 'married', 'married', 'not married', 'not married', 'not married'], inplace = True)
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
        #data['race'].loc[data['race'] != "Caucasian"] = 'Other'
        data['is_med_or_high_risk'] = (data['decile_score'] >= 5).astype(int)
        data['length_of_stay'] = (
                pd.to_datetime(data['c_jail_out']) - pd.to_datetime(data['c_jail_in']))

        #cols = ['age', 'c_charge_degree', 'sex', 'age_cat', 'score_text', 'race', 'priors_count', 'length_of_stay', 'days_b_screening_arrest', 'decile_score', 'two_year_recid']
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

        train = clean_and_encode_dataset(read_dataset(TRAIN_DATA_FILE, data_types, 'adult'), 'adult')  #(32561, 14)
        test = clean_and_encode_dataset(read_dataset(TEST_DATA_FILE, data_types, 'adult'), 'adult')    #(16281, 14)

        train_set_1 = train[train['workclass'] == 1]
        train_set_2 = train[train['workclass'] != 1]

        test_set_1 = test[test['workclass'] == 1]
        test_set_2 = test[test['workclass'] != 1]

        cols = train_set_1.columns

        features, decision = cols[:-1], cols[-1]

        data_train_1 = TabularData(train_set_1[features].values, train_set_1[decision].valuesr)
        data_test_1 = TabularData(test_set_1[features].values, test_set_1[decision].values)

        data_train_2 = TabularData(train_set_2[features].values, train_set_2[decision].values)
        data_test_2 = TabularData(test_set_2[features].values, test_set_2[decision].values)

        train_sets = [train_set_1, train_set_2]
        test_sets = [test_set_1, test_set_2]

        d_train_all = pd.concat(train_sets)
        d_test_all = pd.concat(test_sets)
        data_train_all = TabularData(d_train_all[features].values, d_train_all[decision].values)
        data_test_all = TabularData(d_test_all[features].values, d_test_all[decision].values)

        all_data = [data_train_1, data_train_2, data_test_1, data_test_2]

        return all_data, features, data_train_all, data_test_all

    elif data_name == 'compas':
        train = clean_and_encode_dataset(read_dataset(None, None, 'compas'), 'compas')
        client_1 = train[train['age'] <= 31]  # 3164

        client_2 = train[train['age'] > 31]  # 3008

        cols = client_1.columns

        features, decision = cols[:-1], cols[-1]

        d_train_g_1, d_test_g_1 = train_test_split(client_1, test_size=300)
        d_train_g_2, d_test_g_2 = train_test_split(client_2, test_size=300)

        d_train_g_1 = d_train_g_1.sample(frac=1).reset_index(drop=True)
        d_train_g_2 = d_train_g_2.sample(frac=1).reset_index(drop=True)
        d_test_g_1 = d_test_g_1.sample(frac=1).reset_index(drop=True)
        d_test_g_2 = d_test_g_2.sample(frac=1).reset_index(drop=True)

        train_sets = [d_train_g_1, d_train_g_2]
        test_sets = [d_test_g_1, d_test_g_2]

        d_train_all = pd.concat(train_sets)
        d_test_all = pd.concat(test_sets)


    return d_train_all, d_test_all, len(d_train_g_1), len(d_test_g_1), features, decision


def gen_random_loaders(data_name, num_users, bz, classes_per_user):
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}

    dataloaders = []

    group_all_train_data, group_all_test_data, len_d_train_g_1, len_d_test_g_1, features, decision = get_dataset(data_name, num_users)

    datasets = [group_all_train_data, group_all_test_data]


    for i, d in enumerate(datasets):
        usr_subset_idx = [[] for i in range(num_users)]
        data_copy = d
        if i == 0:
            for usr_i in range(num_users):
                if usr_i == 0 or usr_i == 1:
                    end_idx = int(len_d_train_g_1/ 2)
                if usr_i == 2 or usr_i == 3:
                    end_idx = int(len(data_copy) / 2)

                usr_subset_idx[usr_i].extend(TabularData(data_copy[:end_idx][features].values, data_copy[:end_idx][decision].values))
                data_copy = data_copy[end_idx:]

            subsets = list(usr_subset_idx)
            loader_params['shuffle'] = True
            dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

        elif i == 1:
            for usr_i in range(num_users):
                if usr_i == 0 or usr_i == 1:
                    end_idx = int(len_d_test_g_1 / 2)
                if usr_i == 2 or usr_i == 3:
                    end_idx = int(len(data_copy) / 2)

                usr_subset_idx[usr_i].extend(
                    TabularData(data_copy[:end_idx][features].values, data_copy[:end_idx][decision].values))
                data_copy = data_copy[end_idx:]

            subsets = list(usr_subset_idx)

            dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    return dataloaders, features


