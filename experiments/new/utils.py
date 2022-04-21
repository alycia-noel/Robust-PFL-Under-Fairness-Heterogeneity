import os
import random
import numpy as np
import torch
import logging

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device(no_cuda=False, gpus='4'):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")

def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def TP_FP_TN_FN(x, predicted_prediction, labels_pred, fair):
    TP = [0, 0, 0] # all, f, m
    FP = [0, 0, 0]
    FN = [0, 0, 0]
    TN = [0, 0, 0]

    for i in range(len(x)):
        if fair == 'none':
            if x[i][9].item() == 0:
                if predicted_prediction[i] == 1 and labels_pred[i] == 1:
                    TP[1] += 1
                    TP[0] += 1
                elif predicted_prediction[i] == 1 and labels_pred[i] == 0:
                    FP[1] += 1
                    FP[0] += 1
                elif predicted_prediction[i] == 0 and labels_pred[i] == 0:
                    TN[1] += 1
                    TN[0] += 1
                elif predicted_prediction[i] == 0 and labels_pred[i] == 1:
                    FN[1] += 1
                    FN[0] += 1
            else:
                if predicted_prediction[i] == 1 and labels_pred[i] == 1:
                    TP[2] += 1
                    TP[0] += 1
                elif predicted_prediction[i] == 1 and labels_pred[i] == 0:
                    FP[2] += 1
                    FP[0] += 1
                elif predicted_prediction[i] == 0 and labels_pred[i] == 0:
                    TN[2] += 1
                    TN[0] += 1
                elif predicted_prediction[i] == 0 and labels_pred[i] == 1:
                    FN[2] += 1
                    FN[0] += 1
        else:
            if x[i].item() == 0:
                if predicted_prediction[i] == 1 and labels_pred[i] == 1:
                    TP[1] += 1
                    TP[0] += 1
                elif predicted_prediction[i] == 1 and labels_pred[i] == 0:
                    FP[1] += 1
                    FP[0] += 1
                elif predicted_prediction[i] == 0 and labels_pred[i] == 0:
                    TN[1] += 1
                    TN[0] += 1
                elif predicted_prediction[i] == 0 and labels_pred[i] == 1:
                    FN[1] += 1
                    FN[0] += 1
            else:
                if predicted_prediction[i] == 1 and labels_pred[i] == 1:
                    TP[2] += 1
                    TP[0] += 1
                elif predicted_prediction[i] == 1 and labels_pred[i] == 0:
                    FP[2] += 1
                    FP[0] += 1
                elif predicted_prediction[i] == 0 and labels_pred[i] == 0:
                    TN[2] += 1
                    TN[0] += 1
                elif predicted_prediction[i] == 0 and labels_pred[i] == 1:
                    FN[2] += 1
                    FN[0] += 1

    return TP, FP, TN, FN

def metrics(TP, FP, TN, FN):
    #TP[all, f, m]

    f1_score_prediction = TP[0] / (TP[0] + (FP[0] + FN[0]) / 2)
    f1_female = TP[1] / (TP[1] + (FP[1] + FN[1]) / 2)
    f1_male = TP[2] / (TP[2] + (FP[2] + FN[2]) / 2)

    accuracy = (TP[0] + TN[0]) / (TP[0] + FP[0] + FN[0] + TN[0])
    f_acc = (TP[1] + TN[1]) / (TP[1] + FP[1] + FN[1] + TN[1])
    m_acc = (TP[2] + TN[2]) / (TP[2] + FP[2] + FN[2] + TN[2])


    AOD = (((TP[1] / (TP[1] + FN[1])) - (TP[2] / (TP[2] + FN[2]))) + ((FP[1] / (FP[1] + TN[1])) - (FP[2] / (FP[2] + TN[2])))) / 2  # average odds difference
    EOD = (TP[1] / (TP[1] + FN[1])) - (TP[2] / (TP[2] + FN[2]))                                                                  # equal opportunity difference
    SPD = (TP[1] + FP[1]) / (TP[1] + FP[1] + TN[1] + FN[1]) - (TP[2] + FP[2]) / (TP[2] + FP[2] + TN[2] + FN[2])

    return f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, AOD, EOD, SPD