import random
import os
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class TabularData(Dataset):
    def __init__(self, X, y, s):
        assert len(X) == len(y) == len(s)
        n, m = X.shape
        self.n = n
        self.m = m
        self.X = torch.tensor(X, dtype=torch.float64)
        self.y = torch.tensor(y, dtype=torch.float64)
        self.s = torch.tensor(s, dtype=torch.float64)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.s[idx]

def plot_roc_curves(results, pred_col, resp_col, size=(7, 5), fname=None):
    fpr, tpr, _ = roc_curve(results[resp_col], results[pred_col])
    roc_auc = auc(fpr, tpr)
    return roc_auc

def get_data():
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

    cols = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'priors_count', 'length_of_stay',
            'days_b_screening_arrest', 'decile_score', 'two_year_recid', 'sex']
    compas = df_filtered[cols]

    compas['length_of_stay'] /= np.timedelta64(1, 'D')
    compas['length_of_stay'] = np.ceil(compas['length_of_stay'])

    encoders = {}
    for col in ['race', 'sex', 'c_charge_degree', 'score_text', 'age_cat']:
        encoders[col] = LabelEncoder().fit(compas[col])
        compas.loc[:, col] = encoders[col].transform(compas[col])

    client_1 = compas[compas['age'] <= 31]  # 3164

    client_2 = compas[compas['age'] > 31]  # 3008

    cols_1 = client_1.columns
    features_1, decision_1, sensitive_1 = cols_1[:-2], cols_1[-2], cols_1[-1]

    cols_2 = client_2.columns
    features_2, decision_2, sensitive_2 = cols_2[:-2], cols_2[-2], cols_2[-1]

    d_train_1, d_test_1 = train_test_split(client_1, test_size=300)
    data_train_1 = TabularData(d_train_1[features_1].values, d_train_1[decision_1].values, d_train_1[sensitive_1].values)
    data_test_1 = TabularData(d_test_1[features_1].values, d_test_1[decision_1].values, d_test_1[sensitive_1].values)

    d_train_2, d_test_2 = train_test_split(client_2, test_size=300)
    data_train_2 = TabularData(d_train_2[features_2].values, d_train_2[decision_2].values, d_train_2[sensitive_2].values)
    data_test_2 = TabularData(d_test_2[features_2].values, d_test_2[decision_2].values, d_test_2[sensitive_2].values)

    return data_train_1, data_test_1, data_train_2, data_test_2, features_1, features_2, decision_1, decision_2, d_test_1, d_test_2, encoders

def confusion_matrix(s, predicted_prediction, labels_pred, TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn):
    for i in range(len(s)):
        if s[i].item() == 0:
            if predicted_prediction[i] == 1 and labels_pred[i] == 1:
                f_tp += 1
                TP += 1
            elif predicted_prediction[i] == 1 and labels_pred[i] == 0:
                f_fp += 1
                FP += 1
            elif predicted_prediction[i] == 0 and labels_pred[i] == 0:
                f_tn += 1
                TN += 1
            elif predicted_prediction[i] == 0 and labels_pred[i] == 1:
                f_fn += 1
                FN += 1
        else:
            if predicted_prediction[i] == 1 and labels_pred[i] == 1:
                m_tp += 1
                TP += 1
            elif predicted_prediction[i] == 1 and labels_pred[i] == 0:
                m_fp += 1
                FP += 1
            elif predicted_prediction[i] == 0 and labels_pred[i] == 0:
                m_tn += 1
                TN += 1
            elif predicted_prediction[i] == 0 and labels_pred[i] == 1:
                m_fn += 1
                FN += 1

    return TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn

def metrics(TP, FP, FN, f_tp, f_fp, f_fn, m_tp, m_fp, m_fn, TN, f_tn, m_tn):
    f1_score_prediction = TP / (TP + (FP + FN) / 2)
    f1_female = f_tp / (f_tp + (f_fp + f_fn) / 2)
    f1_male = m_tp / (m_tp + (m_fp + m_fn) / 2)

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    f_acc = (f_tp + f_tn) / (f_tp + f_fp + f_fn + f_tn)
    m_acc = (m_tp + m_tn) / (m_tp + m_fp + m_fn + m_tn)

    error = (FP + FN) / (TP + FP + FN + TN)
    f_err = (f_fp + f_fn) / (f_tp + f_fp + f_fn + f_tn)
    m_err = (m_fp + m_fn) / (m_tp + m_fp + m_fn + m_tn)

    if f_fp == 0 and f_tn == 0:
        AOD = (((f_tp / (f_tp + f_fn)) - (m_tp / (m_tp + m_fn))) + (0 - (m_fp / (m_fp + m_tn)))) / 2
    else:
        AOD = (((f_tp / (f_tp + f_fn)) - (m_tp / (m_tp + m_fn))) + ((f_fp / (f_fp + f_tn)) - (m_fp / (m_fp + m_tn)))) / 2  # average odds difference
    EOD = (f_tp / (f_tp + f_fn)) - (m_tp / (m_tp + m_fn))                                                                  # equal opportunity difference
    SPD = (f_tp + f_fp) / (f_tp + f_fp + f_tn + f_fn) - (m_tp + m_fp) / (m_tp + m_fp + m_tn + m_fn)

    return f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, AOD, EOD, SPD