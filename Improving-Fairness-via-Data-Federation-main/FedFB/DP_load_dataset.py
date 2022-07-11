import numpy as np
from utils import *
import torch

# synthetic
def dataSplit(train_data, test_data, client_split = ((.5, .2), (.3, .4), (.2, .4)), Z = 2):
    if Z == 2:
        z1_idx = train_data[train_data.z == 1].index
        z0_idx = train_data[train_data.z == 0].index

        client1_idx = np.concatenate((z1_idx[:int(client_split[0][0]*len(z1_idx))], z0_idx[:int(client_split[0][1]*len(z0_idx))]))
        client2_idx = np.concatenate((z1_idx[int(client_split[0][0]*len(z1_idx)):int((client_split[0][0] + client_split[1][0])*len(z1_idx))],
                                      z0_idx[int(client_split[0][1]*len(z0_idx)):int((client_split[0][1] + client_split[1][1])*len(z0_idx))]))
        client3_idx = np.concatenate((z1_idx[int((client_split[0][0] + client_split[1][0])*len(z1_idx)):], z0_idx[int((client_split[0][1] + client_split[1][1])*len(z0_idx)):]))
        random.shuffle(client1_idx)
        random.shuffle(client2_idx)
        random.shuffle(client3_idx)

        clients_idx = [client1_idx, client2_idx, client3_idx]
        
    elif Z == 3:
        z_idx, z_len = [], []
        for z in range(3):
            z_idx.append(train_data[train_data.z == z].index)
            z_len.append(len(z_idx[z]))

        clients_idx = []
        a, b = np.zeros(3), np.zeros(3)
        for c in range(4):
            if c > 0:
                a += np.array(client_split[c-1]) * z_len 
            b += np.array(client_split[c]) * z_len
            clients_idx.append(np.concatenate((z_idx[0][int(a[0]):int(b[0])],
                                               z_idx[1][int(a[1]):int(b[1])],
                                               z_idx[2][int(a[2]):int(b[2])])))
            random.shuffle(clients_idx[c])
        
    train_dataset = LoadData(train_data, "y", "z")
    test_dataset = LoadData(test_data, "y", "z")

    synthetic_info = [train_dataset, test_dataset, clients_idx]
    return synthetic_info

def dataGenerate(seed = 432, train_samples = 3000, test_samples = 500, 
                y_mean = 0.6, client_split = ((.5, .2), (.3, .4), (.2, .4)), Z = 2):
    """
    Generate dataset consisting of two sensitive groups.
    """
    ########################
    # Z = 2:
    # 3 clients: 
    #           client 1: %50 z = 1, %20 z = 0
    #           client 2: %30 z = 1, %40 z = 0
    #           client 3: %20 z = 1, %40 z = 0
    ########################
    # 4 clients:
    #           client 1: 50% z = 0, 10% z = 1, 20% z = 2
    #           client 2: 30% z = 0, 30% z = 1, 30% z = 2
    #           client 3: 10% z = 0, 30% z = 1, 30% z = 2
    #           client 4: 10% z = 0, 30% z = 1, 20% z = 2
    ########################
    np.random.seed(seed)
    random.seed(seed)
        
    train_data, test_data = dataSample(train_samples, test_samples, y_mean, Z)
    return dataSplit(train_data, test_data, client_split, Z)

synthetic_info = dataGenerate(seed = 123, test_samples = 1500, train_samples = 3500)

# Adult
sensitive_attributes = ['sex']
categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
features_to_keep = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 
            'native-country', 'salary']
label_name = 'salary'

adult = process_csv('adult', 'adult.data', label_name, ' >50K', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep)
test = process_csv('adult', 'adult.test', label_name, ' >50K.', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep) # the distribution is very different from training distribution
test['native-country_ Holand-Netherlands'] = 0
test = test[adult.columns]

np.random.seed(1)
adult_private_idx = adult[adult['workclass_ Private'] == 1].index
adult_others_idx = adult[adult['workclass_ Private'] == 0].index
adult_mean_sensitive = adult['z'].mean()

client1_idx = np.concatenate((np.random.choice(adult_private_idx, int(.8*len(adult_private_idx)), replace = False),
                                np.random.choice(adult_others_idx, int(.2*len(adult_others_idx)), replace = False)))
client2_idx = np.array(list(set(adult.index) - set(client1_idx)))
adult_clients_idx = [client1_idx, client2_idx]

adult_num_features = len(adult.columns)-1
adult_test = LoadData(test, 'salary', 'z')
adult_train = LoadData(adult, 'salary', 'z')
torch.manual_seed(0)
adult_info = [adult_train, adult_test, adult_clients_idx]

# COMPAS
sensitive_attributes = ['sex', 'race']
categorical_attributes = ['age_cat', 'c_charge_degree', 'c_charge_desc']
continuous_attributes = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
features_to_keep = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'c_charge_degree', 'c_charge_desc','two_year_recid']
label_name = 'two_year_recid'

compas = process_csv('compas', 'compas-scores-two-years.csv', label_name, 0, sensitive_attributes, ['Female', 'African-American'], categorical_attributes, continuous_attributes, features_to_keep)
train = compas.iloc[:int(len(compas)*.7)]
test = compas.iloc[int(len(compas)*.7):]

np.random.seed(1)
torch.manual_seed(0)
client1_idx = train[train.age > 0.1].index 
client2_idx = train[train.age <= 0.1].index
compas_mean_sensitive = train['z'].mean()
compas_z = len(set(compas.z))

clients_idx = [client1_idx, client2_idx]

compas_num_features = len(compas.columns) - 1
compas_train = LoadData(train, label_name, 'z')
compas_test = LoadData(test, label_name, 'z')

compas_info = [compas_train, compas_test, clients_idx]
