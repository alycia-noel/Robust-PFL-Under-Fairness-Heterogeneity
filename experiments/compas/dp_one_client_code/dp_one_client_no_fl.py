import time
import warnings
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import  DataLoader
from fairtorch import DemographicParityLoss, EqualiedOddsLoss
from utils import seed_everything, plot_roc_curves, get_data, confusion_matrix, metrics
from models import LR, NN, LR_context, NN_context
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

m = "neural-net"

no_cuda=False
gpus='3'
device = torch.cuda.set_device(3)

seed_everything(0)

data_train, data_test, features, d_train, d_test, encoders = get_data()

all_acc, all_f_acc, all_m_acc = [], [], []
all_test_error, all_F_ERR, all_M_ERR = [], [], []
all_tp, all_tn, all_fp, all_fn, all_f1 = [], [], [], [], []
all_F_TP, all_F_FP, all_F_TN, all_F_FN, all_F_F1 = [], [], [], [], []
all_M_TP, all_M_FP, all_M_TN, all_M_FN, all_M_F1 = [], [], [], [], []
all_EOD, all_SPD, all_AOD = [], [], []
all_times, all_roc = [], []

for i in range(1):
    seed_everything(0)
    print('Round: ', i)
    if m == "log-reg":
        model = LR(input_size=9)
        lr = .00025        # for dp
        #lr = .0003 #.001         # for eo
        wd = 0
        alpha = 70       # for dp
        #alpha = 5          #for eo
        eps = 150
    elif m == "neural-net":
        model = NN(input_size=9)
        lr = .0005     # for dp
        #lr = .0005
        wd = .0000001
        alpha = 70      # for dp
        #alpha = 15
        eps = 100
    elif m == "log-reg-c":
        model = LR_context(input_size=9, vector_size=9)
        lr = .0009 #5e-4
        wd = 0
        alpha = 1
        eps = 100
    elif m == "neural-net-c":
        model = NN_context(input_size=9, vector_size=9)
        lr = .0005
        wd = 0.0000005
        alpha = 1
        eps = 100

    model = model.double()
    model = model.to(device)

    # for dp
    train_loader = DataLoader(data_train, shuffle = True, batch_size = 128)
    test_loader = DataLoader(data_test, shuffle = False, batch_size= 128)

    # for eo
    #train_loader = DataLoader(data_train, shuffle=True, batch_size=256)
    #test_loader = DataLoader(data_test, shuffle = False, batch_size=256)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
    loss = nn.BCEWithLogitsLoss(reduction='mean') #Binary Cross Entropy loss
    dp_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=alpha)
    #eo_loss = EqualiedOddsLoss(sensitive_classes=[0, 1], alpha=alpha)

    loss_values = []
    test_loss_values = []
    test_error, F_ERR, M_ERR = [], [], []

    results = []

    acc_values, test_acc, F_ACC, M_ACC = [], [], [], []

    tp, tn, fp, fn, f1 = [], [], [], [], []
    F_TP, F_FP, F_TN, F_FN, F_F1 = [], [], [], [], []
    M_TP, M_FP, M_TN, M_FN, M_F1 = [], [], [], [], []
    EOD, SPD, AOD = [], [], []

    times = []

    # Train model

    model.train() #warm-up
    epochs = eps
    start_epoch = time.time()
    for epoch in range(epochs): #original 250, best:700
        running_loss = 0.0
        correct = 0

        for i, (x, y,s ) in enumerate(train_loader):
            optimizer.zero_grad()
            y_, y_raw = model(x.to(device))
            err = loss(y_raw.flatten(), y.to(device)) + dp_loss(x.float(), y_raw.float(), s.float())
            #err = loss(y_raw.flatten(), y.to(device)) + eo_loss(x.float(), y_raw.float(), s.float(), y.float())
            err = err.mean()
            running_loss += err.item() * x.size(0)
            err.backward()
            optimizer.step()

            preds = y_.round().reshape(1, len(y_))
            correct += (preds.eq(y)).sum().item()

        accuracy = (100 * correct / len(data_train))
        acc_values.append(accuracy)
        loss_values.append(running_loss / len(train_loader))

        # Eval Model
        model.eval()
        predictions = []
        running_loss_test = 0
        TP, FP, FN, TN = 0, 0, 0, 0
        f_tp, f_fp, f_tn, f_fn = 0, 0, 0, 0
        m_tp, m_fp, m_tn, m_fn = 0, 0, 0, 0
        correct = 0.0

        with torch.no_grad():
            for i, (x, y, s) in enumerate(test_loader):
                pred, pred_raw = model(x.to(device))
                test_err = loss(pred_raw.flatten(), y.to(device))
                test_err = test_err.mean()
                running_loss_test += test_err.item() * x.size(0)

                preds = pred.round().reshape(1, len(pred))
                predictions.extend(preds.flatten().numpy())
                correct += (preds.eq(y)).sum().item()
                predicted_prediction = preds.type(torch.IntTensor).numpy().reshape(-1)
                labels_pred = y.type(torch.IntTensor).numpy().reshape(-1)

                TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn = confusion_matrix(s, predicted_prediction, labels_pred, TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn)

        test_loss_values.append(running_loss_test / len(test_loader))

        f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, aod, eod, spd = metrics(
            TP, FP, FN, f_tp, f_fp, f_fn, m_tp, m_fp, m_fn, TN, f_tn, m_tn)

        f1.append(f1_score_prediction)
        F_F1.append(f1_female)
        M_F1.append(f1_male)
        AOD.append(aod)
        SPD.append(spd)
        EOD.append(eod)

        res = (
            pd.DataFrame(columns=features, index=d_test.index)
                .add_suffix('_partial')
                .join(d_test)
                .assign(prediction=predictions)
                .assign(round=epoch)
        )

        results.append(res)
        test_acc.append(accuracy)
        F_ACC.append(f_acc)
        M_ACC.append(m_acc)
        test_error.append(error)
        F_ERR.append(f_err)
        M_ERR.append(m_err)

        if epoch == epochs - 1:
            plt.plot(loss_values, label='Train Loss')
            plt.plot(test_loss_values, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            title_loss = 'Loss over Epochs'
            plt.title(title_loss)
            plt.legend(loc="upper right")
            plt.show()

        tn.append(TN)
        tp.append(TP)
        fn.append(FN)
        fp.append(FP)

        F_TN.append(f_tn)
        F_TP.append(f_tp)
        F_FN.append(f_fn)
        F_FP.append(f_fp)

        M_TN.append(m_tn)
        M_TP.append(m_tp)
        M_FN.append(m_fn)
        M_FP.append(m_fp)

    results = pd.concat(results)
    average_test_acc = sum(test_acc) / len(test_acc)
    female_test_acc = sum(F_ACC) / len(F_ACC)
    male_test_acc = sum(M_ACC) / len(M_ACC)

    all_acc.append(test_acc[len(test_acc) - 1])
    all_f_acc.append(F_ACC[len(F_ACC) - 1])
    all_m_acc.append(M_ACC[len(M_ACC) - 1])
    all_test_error.append(test_error[len(test_error) - 1])
    all_F_ERR.append(F_ERR[len(F_ERR) - 1])
    all_M_ERR.append(M_ERR[len(M_ERR) - 1])
    all_tp.append(tp[len(tp) - 1])
    all_tn.append(tn[len(tn) - 1])
    all_fp.append(fp[len(fp) - 1])
    all_fn.append(fn[len(fn) - 1])
    all_f1.append(f1[len(f1) - 1])
    all_F_TP.append(F_TP[len(F_TP) - 1])
    all_F_FP.append(F_FP[len(F_FP) - 1])
    all_F_TN.append(F_TN[len(F_TN) - 1])
    all_F_FN.append(F_FN[len(F_FN) - 1])
    all_F_F1.append(F_F1[len(F_F1) - 1])
    all_M_TP.append(M_TP[len(M_TP) - 1])
    all_M_FP.append(M_FP[len(M_FP) - 1])
    all_M_TN.append(M_TN[len(M_TN) - 1])
    all_M_FN.append(M_FN[len(M_FN) - 1])
    all_M_F1.append(M_F1[len(M_F1) - 1])
    all_EOD.append(EOD[len(EOD) - 1])
    all_SPD.append(SPD[len(SPD) - 1])
    all_AOD.append(AOD[len(AOD) - 1])
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    times.append(elapsed)
    total_time = sum(times)
    all_times.append(total_time)

    for col, encoder in encoders.items():
        results.loc[:, col] = encoder.inverse_transform(results[col])

    all_roc.append(plot_roc_curves(results, 'prediction', 'two_year_recid', size=(7, 5), fname='./results/roc.png'))

print('*******************')
print('*       all       *')
print('*******************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_acc)))
print('Test Error: {0:1.3f}'.format(np.mean(all_test_error)))
print(
    'TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_tp), np.mean(all_fp), np.mean(all_tn),
                                                                      np.mean(all_fn)))
print('EOD: {0:1.4f}'.format(np.mean(all_EOD)))
print('SPD: {0:1.4f}'.format(np.mean(all_SPD)))
print('AOD: {0:1.4f}'.format(np.mean(all_AOD)))
print('F1: {0:1.3f}'.format(np.mean(all_f1)))
print("")
print('**********************')
print('*       Female       *')
print('**********************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_f_acc)))
print('Test Error: {0:1.3f}'.format(np.mean(all_F_ERR)))
print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_F_TP), np.mean(all_F_FP),
                                                                        np.mean(all_F_TN), np.mean(all_F_FN)))
print('F1: {0:1.3f}'.format(np.mean(all_F_F1)))
print("")
print('********************')
print('*       Male       *')
print('********************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_m_acc)))
print('Test Error: {0:1.3f}'.format(np.mean(all_M_ERR)))
print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_M_TP), np.mean(all_M_FP),
                                                                        np.mean(all_M_TN), np.mean(all_M_FN)))
print('F1: {0:1.3f}'.format(np.mean(all_M_F1)))

print('Train Time: {0:1.2f}'.format(np.mean(all_times)))
print('AUROC: {0:1.2f}'.format(np.mean(all_roc)))

