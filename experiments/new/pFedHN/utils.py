import os
import random
import numpy as np
import torch
import logging

class AscentFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_input):
        return -grad_input

def make_ascent(loss):
    return AscentFunction.apply(loss)

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

def metrics(TP, FP, TN, FN): #double checked, all metrics are calculated correctly based on the aif360
    blank = [0, 0, 0]

    #TP[all, f, m]
    if TP == blank:
        f1_score_prediction, f1_female, f1_male = 0, 0, 0
    elif TP[1] == 0 and FP[1] == 0:
        f1_female = 0
        precision = TP[0] / (TP[0] + FP[0])
        precision_m = TP[2] / (TP[2] + FP[2])

        recall = TP[0] / (TP[0] + FN[0])
        recall_m = TP[2] / (TP[2] + FN[2])

        f1_score_prediction = (2 * precision * recall) / (precision + recall)
        f1_male = (2 * precision_m * recall_m) / (precision_m + recall_m)
    elif TP[2] == 0 and FP[2] == 0:
        f1_male = 0
        precision = TP[0] / (TP[0] + FP[0])
        precision_f = TP[1] / (TP[1] + FP[1])

        recall = TP[0] / (TP[0] + FN[0])
        recall_f = TP[1] / (TP[1] + FN[1])

        f1_score_prediction = (2 * precision * recall) / (precision + recall)
        f1_female = (2 * precision_f * recall_f) / (precision_f + recall_f)
    else:
        precision = TP[0] / (TP[0] + FP[0])
        precision_f = TP[1] / (TP[1] + FP[1])
        precision_m = TP[2] / (TP[2] + FP[2])

        recall = TP[0] / (TP[0] + FN[0])
        recall_f = TP[1] / (TP[1] + FN[1])
        recall_m = TP[2] / (TP[2] + FN[2])

        if TP[1] != 0:
            f1_female = (2 * precision_f * recall_f) / (precision_f + recall_f)
        else:
            f1_female = 0

        if TP[2] != 0:
            f1_male = (2 * precision_m * recall_m) / (precision_m + recall_m)
        else:
            f1_male = 0

        f1_score_prediction = (2 * precision * recall ) / (precision + recall)

    accuracy = (TP[0] + TN[0]) / (TP[0] + FP[0] + FN[0] + TN[0])
    f_acc = (TP[1] + TN[1]) / (TP[1] + FP[1] + FN[1] + TN[1])
    m_acc = (TP[2] + TN[2]) / (TP[2] + FP[2] + FN[2] + TN[2])


    aod = (((TP[1] / (TP[1] + FN[1])) - (TP[2] / (TP[2] + FN[2]))) + ((FP[1] / (FP[1] + TN[1])) - (FP[2] / (FP[2] + TN[2])))) * .5  # average odds difference
    eod = (TP[1] / (TP[1] + FN[1])) - (TP[2] / (TP[2] + FN[2]))                                                                     # equal opportunity difference
    spd = (TP[1] + FP[1]) / (TP[1] + FP[1] + TN[1] + FN[1]) - (TP[2] + FP[2]) / (TP[2] + FP[2] + TN[2] + FN[2])

    return f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, aod, eod, spd
