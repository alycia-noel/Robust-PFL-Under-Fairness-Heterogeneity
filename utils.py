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

def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def TP_FP_TN_FN(x, predicted_prediction, labels_pred, which_position):

    TP = [0, 0, 0] # all, f, m
    FP = [0, 0, 0]
    FN = [0, 0, 0]
    TN = [0, 0, 0]

    for i in range(len(x)):
        if x[i][which_position].item() == 0:
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
    try:
        eod = .5 * (((FP[1] / (FP[1] + TN[1])) - (FP[2] / (FP[2] + TN[2]))) + (
                    (TP[1] / (TP[1] + FN[1])) - (TP[2] / (TP[2] + FN[2]))))
    except ZeroDivisionError:
        eod = 0

    accuracy = (TP[0] + TN[0]) / (TP[0] + FP[0] + FN[0] + TN[0])
    spd = (TP[1] + FP[1]) / (TP[1] + FP[1] + TN[1] + FN[1]) - (TP[2] + FP[2]) / (TP[2] + FP[2] + TN[2] + FN[2])

    return accuracy, eod, spd
