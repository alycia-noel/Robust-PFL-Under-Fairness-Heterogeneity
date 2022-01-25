# This file handles the data download, parsing, and preparation
# It is configured for images (CIFAR10 and CIFAR 100)
# TODO:
    # Need to add capability for tabular data

# Imports
import random
from collections import defaultdict

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

# This function returns the needed data splits for training, validating, and testing for CIFAR10 and CIFAR100
# data_name: name of dataset, choose from [cifar10, cifar100]
# dataroot: root to data dir
# normalize: True/False to normalize the data [default is true]
# val_size: validation split size (in #samples, default is 10,000)
# return: train_set, val_set, test_set (tuple of pytorch dataset/subset)

def get_datasets(data_name, dataroot, normalize=True, val_size=10000):

    # set normalization function and data_objects for the datasets
    if data_name =='cifar10':
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
    elif data_name == 'cifar100':
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100
    else:
        raise ValueError("choose data_name from ['mnist', 'cifar10', 'cifar100']")

    # Start of transformation function
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

def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.6, low_prob=0.4):
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
# dataname: name of dataset [can choose from cifar10 or cifar100)
# data_path: the root path for the data directory
# num_users: the number of clients
# bz: the batch size
# classes_per_user: number of classes assigned to each client
# returns the train/val/test loaders of each client, so you get a list of pytorch loaders
def gen_random_loaders(data_name, data_path, num_users, bz, classes_per_user):
    loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0}
    dataloaders = []
    datasets = get_datasets(data_name, data_path, normalize=True)
    for i, d in enumerate(datasets):
        # ensure same partition for train/test/val
        if i == 0:
            cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
            loader_params['shuffle'] = True
        usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
        # create subsets for each client
        subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
        # create dataloaders from subsets
        dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

    return dataloaders
