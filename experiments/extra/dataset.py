# This file handles the data download, parsing, and preparation

# Imports
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import warnings
warnings.filterwarnings("ignore")

class TabularData(Dataset):
    def __init__(self, X, y, train=True, download=False, transform=None):
        assert len(X) == len(y)
        n, m = X.shape
        self.n = n
        self.m = m
        self.X = torch.tensor(X, dtype=torch.float64)
        self.targets = torch.tensor(y, dtype=torch.float64)
        self.transform = transform
        self.train = train
        self.download = download

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        img = self.X[index]
        target = self.targets[index]
        return img, target

# This function returns the needed data splits for training, validating, and testing
# data_name: name of dataset, choose from [cifar10, cifar100, COMPAS]
# dataroot: root to data dir
# normalize: True/False to normalize the data [default is true]
# val_size: validation split size
# return: train_set, val_set, test_set (tuple of pytorch dataset/subset)

def get_datasets(data_name, dataroot, normalize, val_size):

    # set normalization function and data_objects for the datasets
    if data_name =='cifar10':
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
    elif data_name == 'cifar100':
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100
    elif data_name == 'COMPAS':
        url = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
        df = pd.read_csv(url)
        df_filtered = df.loc[df['days_b_screening_arrest'] <= 30]
        df_filtered = df_filtered.loc[df_filtered['days_b_screening_arrest'] >= -30]
        df_filtered = df_filtered.loc[df_filtered['is_recid'] != -1]
        df_filtered = df_filtered.loc[df_filtered['c_charge_degree'] != "O"]
        df_filtered = df_filtered.loc[df_filtered['score_text'] != 'N/A']
        df_filtered['is_med_or_high_risk'] = (df_filtered['decile_score'] >= 5).astype(int)
        df_filtered['length_of_stay'] = (
                    pd.to_datetime(df_filtered['c_jail_out']) - pd.to_datetime(df_filtered['c_jail_in']))

        cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'length_of_stay',
                'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid']
        compas = df_filtered[cols]

        compas['length_of_stay'] /= np.timedelta64(1, 'D')
        compas['length_of_stay'] = np.ceil(compas['length_of_stay'])

        encoders = {}
        for col in ['race', 'sex', 'c_charge_degree', 'score_text', 'age_cat']:
            encoders[col] = LabelEncoder().fit(compas[col])
            compas.loc[:, col] = encoders[col].transform(compas[col])

        client_1 = compas[compas['age'] <= 31] #3164

        client_2 = compas[compas['age'] > 31] #3008

        cols_1 = client_1.columns
        features_1, decision_1 = cols_1[:-1], cols_1[-1]

        cols_2 = client_2.columns
        features_2, decision_2 = cols_2[:-1], cols_2[-1]
    else:
        raise ValueError("choose data_name from ['COMPAS', 'cifar10', 'cifar100']")

    # Start of transformation function
    if data_name != 'COMPAS':
        trans = [transforms.ToTensor()]

        # If normalizing, first transform the data to a tensor then normalize it
        if normalize:
            trans.append(normalization)

        # Full transformation function
        transform = transforms.Compose(trans)

        # Build the train+val dataset
        dataset = data_obj(
            dataroot,
            train=True,
            download=True,
            transform=transform
        )

        # Build the test dataset
        test_set = data_obj(
            dataroot,
            train=False,
            download=True,
            transform=transform
        )

        # Separate dataset into the train set and val set
        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Return the datasets
        return train_set, val_set, test_set

    if data_name == 'COMPAS':
        d_train_1, d_test_1 = train_test_split(client_1, test_size=300)
        data_train_1 = TabularData(d_train_1[features_1].values, d_train_1[decision_1].values, train=True, download=False, transform=None)
        test_set_1 = TabularData(d_test_1[features_1].values, d_test_1[decision_1].values, train=False, download=False, transform=None)
        train_set_1, val_set_1 = torch.utils.data.random_split(data_train_1, [2564, 300])#2364, 300]) should be 5172, minused three to test the one client case

        d_train_2, d_test_2 = train_test_split(client_2, test_size=300)
        data_train_2 = TabularData(d_train_2[features_2].values, d_train_2[decision_2].values, train=True, download=False, transform=None)
        test_set_2 = TabularData(d_test_2[features_2].values, d_test_2[decision_2].values, train=False, download=False, transform=None)
        train_set_2, val_set_2 = torch.utils.data.random_split(data_train_2, [2408, 300])

        return train_set_1, train_set_2, val_set_1, val_set_2, test_set_1, test_set_2
        #return train_set_1, val_set_1, test_set_1
# This function pulls relevant information about the dataset such as the number of classes, number of samples,
# and a list of data labels
# dataset: the pytorch dataset object

def get_num_classes_samples(dataset):
    # ---------------#
    # Extract labels #
    # ---------------#
    # has handling for if the data is in list form or not
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets

    # Gets the number of classes and how many samples are in each class
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)

    # Return the gathered information
    return num_classes, num_samples, data_labels_list

# Creates the data distribution for each client on their data. I.e., this function generates the splits for the data
# of all the different clients
# dataset: pytorch dataset object
# num_users: the number of clients
# classes_per_user: number of classes assigned to each client (default = 2)
# high_prob: highest probability sampled (default = 0.6)
# low_prob: lowest probability sampled (default = 0.4)
# return: a dictionary mapping between the classes and proportions for a client. Each entry in the dictionary is
#         a different client

def gen_classes_per_node(dataset, num_users, classes_per_user=1, high_prob=0.6, low_prob=0.4):
    # Get the number of classes and number of samples per class of the entire dataset
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes # how many items per class a client gets
    class_dict = {}
    for i in range(num_classes):
        # sampling alpha_i_c, size = output shape (i.e., we will get count_per_class amount of probabilities)
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing the probability
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)

    # for each user
    for i in range(num_users):
        c = []
        # for each class the user has
        for _ in range(classes_per_user):
            # get the number of samples per class
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            # get an array of the indicies that contain the maximum of the class counts
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            # append a single random sample of the max_class_counts indicies to c
            c.append(np.random.choice(max_class_counts))
            # from the element at the end of the c list, find that class in class_dict and minus 1
            class_dict[c[-1]]['count'] -= 1

        # add c to the class_partition list
        class_partitions['class'].append(c)
        # add the class probabilities to the class_partition list
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])

    # Return the wanted distribution of data to the clients
    return class_partitions

# This function actually divides the data (specifically the data indicies) to each client based on class_partition
# dataset: pytorch dataset object (this will be a train/val/test set)
# num_users: the number of clients
# class_partitions: proportion of class per client
# returns: a dictionary mapping for each client to their data

def gen_data_split(dataset, num_users, class_partitions):

    # Get the number of classes, number of samples per class, and the labels going with the data
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx

# This function generates the train/validate/test loaders for each client
# dataname: name of dataset [can choose from cifar10, cifar100, compas)
# data_path: the root path for the data directory
# num_users: the number of clients
# bz: the batch size
# classes_per_user: number of classes assigned to each client
# returns the train/val/test loaders of each client, so you get a list of pytorch loaders
def gen_random_loaders(data_name, data_path, num_users, bz, classes_per_user):
    bz = 32
    loader_params = {"batch_size": bz, "shuffle": True, "pin_memory": True, "num_workers": 0}
    loader_params_true = {"batch_size": bz, "shuffle": True, "pin_memory": True, "num_workers": 0}
    loader_params_false = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}

    dataloaders = []
    datasets = get_datasets(data_name, data_path, normalize=False, val_size=500)

    # if data_name == 'cifar10' or 'cifar100':
    #     datasets = get_datasets(data_name, data_path, normalize=False, val_size=500)
    #     for i, d in enumerate(datasets):  # train, val, test
    #         # ensure same partition for train/test/val
    #         if i == 0:
    #             cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
    #             loader_params['shuffle'] = True
    #         usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
    #         # create subsets for each client
    #         subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
    #         # create dataloaders from subsets
    #         dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    if data_name == 'COMPAS':
        dataloaders = []
        train_set_1, train_set_2, val_set_1, val_set_2, test_set_1, test_set_2 = datasets
        #train_set_1, val_set_1, test_set_1 = datasets
        train_sets = [train_set_1, train_set_2]
        val_sets = [val_set_1, val_set_2]
        test_loaders = [test_set_1, test_set_2]
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params_true), train_sets)))
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params_false), val_sets)))
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params_false), test_loaders)))

        # dataloaders.append(torch.utils.data.DataLoader(train_set_1, **loader_params_true))
        # dataloaders.append(torch.utils.data.DataLoader(val_set_1, **loader_params_false))
        # dataloaders.append(torch.utils.data.DataLoader(test_set_1, **loader_params_false))
    return dataloaders
