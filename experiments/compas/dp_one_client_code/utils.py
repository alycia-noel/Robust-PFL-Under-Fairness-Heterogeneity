from torch.utils.data import Dataset
import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
            'days_b_screening_arrest', 'decile_score', 'two_year_recid', 'sex', ]
    compas = df_filtered[cols]

    compas['length_of_stay'] /= np.timedelta64(1, 'D')
    compas['length_of_stay'] = np.ceil(compas['length_of_stay'])

    cols = compas.columns
    features, decision, sensitive = cols[:-2], cols[-2], cols[-1]

    encoders = {}
    for col in ['race', 'sex', 'c_charge_degree', 'score_text', 'age_cat']:
        encoders[col] = LabelEncoder().fit(compas[col])
        compas.loc[:, col] = encoders[col].transform(compas[col])

    d_train, d_test = train_test_split(compas, test_size=617)

    data_train = TabularData(d_train[features].values, d_train[decision].values, d_train[sensitive].values)
    data_test = TabularData(d_test[features].values, d_test[decision].values, d_test[sensitive].values)

    return data_train, data_test, features, d_train, d_test, encoders

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