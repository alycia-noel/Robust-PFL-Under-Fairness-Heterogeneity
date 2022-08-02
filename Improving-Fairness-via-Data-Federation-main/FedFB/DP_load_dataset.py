import numpy as np
from utils import *
import torch

# Adult
sensitive_attributes = ['sex']
categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
features_to_keep = ['age', 'workclass', 'education', 'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss','hours_per_week',
            'native_country', 'salary']
all_cols = ['age', 'workclass', 'fnlwegt', 'education', 'education_num', 'marital_status',
            'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss','hours_per_week',
            'native_country', 'salary']
label_name = 'salary'

train = process_csv('adult', 'adult.data', label_name, ' >50K', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = all_cols)
test = process_csv('adult', 'adult.test', label_name, ' >50K.', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = all_cols)

train_data = train.sort_values('workclass').reset_index(drop=True)
test_data = test.sort_values('workclass').reset_index(drop=True)
splits = [train_data.index[np.searchsorted(train_data['workclass'], .5, side='right')],test_data.index[np.searchsorted(test_data['workclass'], .5, side='right')+1]]
datasets = [train_data, test_data]

cols = train_data.columns
features, labels = cols[:-1], cols[-1]

adult_num_features = len(train.columns)-1

adult_info = [train_data, test_data, splits]

# COMPAS
# sensitive_attributes = ['sex', 'race']
# categorical_attributes = ['age_cat', 'c_charge_degree', 'c_charge_desc']
# continuous_attributes = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
# features_to_keep = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
#         'priors_count', 'c_charge_degree', 'c_charge_desc','two_year_recid']
# label_name = 'two_year_recid'
#
# compas = process_csv('compas', 'compas-scores-two-years.csv', label_name, 0, sensitive_attributes, ['Female', 'African-American'], categorical_attributes, continuous_attributes, features_to_keep)
# train = compas.iloc[:int(len(compas)*.7)]
# test = compas.iloc[int(len(compas)*.7):]
#
# np.random.seed(1)
# torch.manual_seed(0)
# client1_idx = train[train.age > 0.1].index
# client2_idx = train[train.age <= 0.1].index
# compas_mean_sensitive = train['z'].mean()
# compas_z = len(set(compas.z))
#
# clients_idx = [client1_idx, client2_idx]
#
# compas_num_features = len(compas.columns) - 1
# compas_train = LoadData(train, label_name, 'z')
# compas_test = LoadData(test, label_name, 'z')
#
# compas_info = [compas_train, compas_test, clients_idx]
