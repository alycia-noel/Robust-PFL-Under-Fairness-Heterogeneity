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

class TabularData(Dataset):
    def __init__(self, X, y, s, fair):
        assert len(X) == len(y)
        n, m = X.shape
        self.n = n
        self.m = m
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        self.fair = fair
        if fair != 'none':
            self.s = torch.tensor(s, dtype=torch.float64)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.fair == 'none':
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx], self.s[idx]


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

def clean_and_encode_dataset(data, data_name, fair):
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
        data['is_med_or_high_risk'] = (data['decile_score'] >= 5).astype(int)
        data['length_of_stay'] = (
                pd.to_datetime(data['c_jail_out']) - pd.to_datetime(data['c_jail_in']))

        if fair == 'none':
            cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'length_of_stay',
                    'days_b_screening_arrest', 'decile_score', 'two_year_recid']
        else:
            cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'priors_count', 'length_of_stay',
                    'days_b_screening_arrest', 'decile_score', 'two_year_recid', 'sex']

        data = data[cols]

        data['length_of_stay'] /= np.timedelta64(1, 'D')
        data['length_of_stay'] = np.ceil(data['length_of_stay'])

        encoders = {}
        for col in ['race', 'sex', 'c_charge_degree', 'score_text', 'age_cat']:
            encoders[col] = LabelEncoder().fit(data[col])
            data.loc[:, col] = encoders[col].transform(data[col])
    return data

def get_dataset(data_name, fair):
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

        train = clean_and_encode_dataset(read_dataset(TRAIN_DATA_FILE, data_types, 'adult'), 'adult', fair)  #(32561, 14)
        test = clean_and_encode_dataset(read_dataset(TEST_DATA_FILE, data_types, 'adult'), 'adult', fair)    #(16281, 14)

        train_set_1 = train[train['workclass'] == 1]
        train_set_2 = train[train['workclass'] != 1]

        test_set_1 = test[test['workclass'] == 1]
        test_set_2 = test[test['workclass'] != 1]

        cols = train_set_1.columns
        if fair == 'none':
            features, decision, sensitive = cols[:-1], cols[-1], None
        else:
            features, decision, sensitive = cols[:-2], cols[-2], cols[-1]

        data_train_1 = TabularData(train_set_1[features].values, train_set_1[decision].values, None, fair)
        data_test_1 = TabularData(test_set_1[features].values, test_set_1[decision].values, None, fair)

        data_train_2 = TabularData(train_set_2[features].values, train_set_2[decision].values, None, fair)
        data_test_2 = TabularData(test_set_2[features].values, test_set_2[decision].values, None, fair)

        train_sets = [train_set_1, train_set_2]
        test_sets = [test_set_1, test_set_2]

        d_train_all = pd.concat(train_sets)
        d_test_all = pd.concat(test_sets)
        data_train_all = TabularData(d_train_all[features].values, d_train_all[decision].values,
                                     None, fair)
        data_test_all = TabularData(d_test_all[features].values, d_test_all[decision].values,
                                    None, fair)

        all_data = [data_train_1, data_train_2, data_test_1, data_test_2]

        return all_data, features, data_train_all, data_test_all, sensitive

    elif data_name == 'compas':
        train = clean_and_encode_dataset(read_dataset(None, None, 'compas'), 'compas', fair)
        client_1 = train[train['age'] <= 31]  # 3164

        client_2 = train[train['age'] > 31]  # 3008

        cols = client_1.columns
        if fair == 'none':
            features, decision, sensitive = cols[:-1], cols[-1], None
        else:
            features, decision, sensitive = cols[:-2], cols[-2], cols[-1]

        if fair == 'none':
            sensitive = None
            d_train_1, d_test_1 = train_test_split(client_1, test_size=300)
            data_train_1 = TabularData(d_train_1[features].values, d_train_1[decision].values, None, fair)
            data_test_1 = TabularData(d_test_1[features].values, d_test_1[decision].values, None, fair)

            d_train_2, d_test_2 = train_test_split(client_2, test_size=300)
            data_train_2 = TabularData(d_train_2[features].values, d_train_2[decision].values, None, fair)
            data_test_2 = TabularData(d_test_2[features].values, d_test_2[decision].values, None, fair)

            train_sets = [d_train_1, d_train_2]
            test_sets = [d_test_1, d_test_2]

            d_train_all = pd.concat(train_sets)
            d_test_all = pd.concat(test_sets)
            data_train_all = TabularData(d_train_all[features].values, d_train_all[decision].values, None, fair)
            data_test_all = TabularData(d_test_all[features].values, d_test_all[decision].values, None, fair)

        else:
            d_train_1, d_test_1 = train_test_split(client_1, test_size=300)
            data_train_1 = TabularData(d_train_1[features].values, d_train_1[decision].values, d_train_1[sensitive].values, fair)
            data_test_1 = TabularData(d_test_1[features].values, d_test_1[decision].values, d_test_1[sensitive].values, fair)

            d_train_2, d_test_2 = train_test_split(client_2, test_size=300)
            data_train_2 = TabularData(d_train_2[features].values, d_train_2[decision].values, d_train_2[sensitive].values, fair)
            data_test_2 = TabularData(d_test_2[features].values, d_test_2[decision].values, d_test_2[sensitive].values, fair)

            train_sets = [d_train_1, d_train_2]
            test_sets = [d_test_1, d_test_2]

            d_train_all = pd.concat(train_sets)
            d_test_all = pd.concat(test_sets)
            data_train_all = TabularData(d_train_all[features].values, d_train_all[decision].values, d_train_all[sensitive].values, fair)
            data_test_all = TabularData(d_test_all[features].values, d_test_all[decision].values, d_test_all[sensitive].values, fair)

        all_data = [data_train_1, data_train_2, data_test_1, data_test_2]

    return all_data, features, data_train_all, data_test_all, sensitive

def get_num_classes_samples(dataset, dataset_split_point_start, dataset_split_point_end):
    data_labels_list = np.array(dataset.y)
    data_labels_list = data_labels_list[dataset_split_point_start:dataset_split_point_end]
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list

def gen_classes_per_node(dataset, num_users, classes_per_user, dataset_split_point_start, dataset_split_point_end, high_prob=0.6, low_prob=0.4):
    num_classes, num_samples, _ = get_num_classes_samples(dataset, dataset_split_point_start, dataset_split_point_end)

    assert (classes_per_user * num_users) % num_classes == 0
    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes):
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions

def gen_data_split(dataset, num_users, class_partitions, dataset_split_point_start, dataset_split_point_end):
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset,dataset_split_point_start, dataset_split_point_end)
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]
    return user_data_idx

def gen_random_loaders(data_name, num_users, bz, classes_per_user, fair):
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}

    num_users_1 = random.randint(1, num_users-1)
    num_users_2 = abs(num_users - num_users_1)

    dataloaders = []
    subsets_train = []
    subsets_test = []
    cls_partitions_1 = []
    cls_partitions_2 = []
    datasets, features, all_train, all_test, sensitive = get_dataset(data_name, fair)

    for i, d in enumerate(datasets):
        if i == 0:
            dataset_split_point_start = 0
            dataset_split_point_end = len(datasets[0])  # 2864
            num_users = num_users_1
            cls_partitions_1 = gen_classes_per_node(all_train, num_users, classes_per_user, dataset_split_point_start, dataset_split_point_end)
            loader_params['shuffle'] = True
            user_subset_idx_1 = gen_data_split(all_train, num_users, cls_partitions_1, dataset_split_point_start, dataset_split_point_end)
            subsets_train = list(map(lambda x: torch.utils.data.Subset(all_train, x), user_subset_idx_1))

        elif i == 1:
            dataset_split_point_start = len(datasets[0])  # 2864
            dataset_split_point_end= len(all_train)
            num_users = num_users_2
            cls_partitions_2 = gen_classes_per_node(all_train, num_users, classes_per_user, dataset_split_point_start, dataset_split_point_end)
            loader_params['shuffle'] = True
            user_subset_idx_2 = gen_data_split(all_train, num_users, cls_partitions_2,dataset_split_point_start, dataset_split_point_end)
            for j in range(num_users):
                subsets_train.append(torch.utils.data.Subset(all_train, user_subset_idx_2[j]))
            dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets_train)))

        if i == 2:
            dataset_split_point_start = 0
            dataset_split_point_end = len(datasets[2])  # 2864
            user_subset_idx_1 = gen_data_split(all_test, num_users, cls_partitions_1, dataset_split_point_start, dataset_split_point_end)
            subsets_test = list(map(lambda x: torch.utils.data.Subset(all_test, x), user_subset_idx_1))

        elif i == 3:
            dataset_split_point_start = len(datasets[2])  # 2864
            dataset_split_point_end= len(all_test)
            user_subset_idx_2 = gen_data_split(all_test, num_users, cls_partitions_2,dataset_split_point_start, dataset_split_point_end)
            for j in range(num_users):
                subsets_test.append(torch.utils.data.Subset(all_test, user_subset_idx_2[j]))
            dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets_test)))



    return dataloaders, features, sensitive


