import time
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import seed_everything, plot_roc_curves, get_data, confusion_matrix, metrics
from models import LR, NN, LR_context, NN_context

warnings.filterwarnings("ignore")

m = "neural-net-c-two"

no_cuda=False
gpus='4'
device = torch.cuda.set_device(4)

seed_everything(0)

data_train_1, data_test_1, data_train_2, data_test_2, features_1, features_2, decision_1, decision_2, d_test_1, d_test_2, encoders = get_data()

all_acc_1, all_f_acc_1, all_m_acc_1 = [], [], []
all_test_error_1, all_F_ERR_1, all_M_ERR_1 = [], [], []
all_tp_1, all_tn_1, all_fp_1, all_fn_1, all_f1_1 = [], [], [], [], []
all_F_TP_1, all_F_FP_1, all_F_TN_1, all_F_FN_1, all_F_F1_1 = [], [], [], [], []
all_M_TP_1, all_M_FP_1, all_M_TN_1, all_M_FN_1, all_M_F1_1 = [], [], [], [], []
all_EOD_1, all_SPD_1, all_AOD_1 = [], [], []
all_times_1, all_roc_1 = [], []

all_acc_2, all_f_acc_2, all_m_acc_2 = [], [], []
all_test_error_2, all_F_ERR_2, all_M_ERR_2 = [], [], []
all_tp_2, all_tn_2, all_fp_2, all_fn_2, all_f1_2 = [], [], [], [], []
all_F_TP_2, all_F_FP_2, all_F_TN_2, all_F_FN_2, all_F_F1_2 = [], [], [], [], []
all_M_TP_2, all_M_FP_2, all_M_TN_2, all_M_FN_2, all_M_F1_2 = [], [], [], [], []
all_EOD_2, all_SPD_2, all_AOD_2 = [], [], []
all_times_2, all_roc_2 = [], []

clients = 2

for i in range(10):
    print('Round: ', i)

    if m == "log-reg-two":
        model = LR(input_size=10)
        l = 2.e-4
    elif m == "neural-net-two":
        model = NN(input_size=10)
        l = .0001
    elif m == "log-reg-c-two":
        model = LR_context(input_size=10, vector_size=10)
        l = 5.e-4
    elif m == "neural-net-c-two":
        model = NN_context(input_size=10, vector_size=10)
        l = .0001

    model = model.double()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = l)
    loss = nn.BCELoss(reduction='mean')

    for c in range(clients):

        if c == 0:
            data_train = data_train_1
            data_test = data_test_1
        else:
            data_train = data_train_2
            data_test = data_test_2

        train_loader = DataLoader(data_train, shuffle = True, batch_size = 128)
        test_loader = DataLoader(data_test, shuffle = False, batch_size= 128)

        results = []
        loss_values = []
        test_loss_values = []
        test_error, F_ERR, M_ERR = [], [], []

        acc_values, test_acc, F_ACC, M_ACC = [], [], [], []

        tp, tn, fp, fn, f1 = [], [], [], [], []
        F_TP, F_FP, F_TN, F_FN, F_F1 = [], [], [], [], []
        M_TP, M_FP, M_TN, M_FN, M_F1 = [], [], [], [], []
        EOD, SPD, AOD = [], [], []

        times = []

        # Train model
        model.train() #warm-up
        epochs = 100
        start = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0.0
       
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                y_ = model(x.to(device))
                err = loss(y_.flatten(), y.to(device))
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
                for i, (x, y) in enumerate(test_loader):
                    pred = model(x)
                    test_err = loss(pred.flatten(), y)
                    test_err = test_err.mean()
                    running_loss_test += test_err.item() * x.size(0)

                    preds = pred.round().reshape(1, len(pred))
                    predictions.extend(preds.flatten().numpy())
                    correct += (preds.eq(y)).sum().item()

                    predicted_prediction = preds.type(torch.IntTensor).numpy().reshape(-1)
                    labels_pred = y.type(torch.IntTensor).numpy().reshape(-1)

                    TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn = confusion_matrix(x,
                                                                                                      predicted_prediction,
                                                                                                      labels_pred, TP,
                                                                                                      FP, FN, TN, f_tp,
                                                                                                      f_fp, f_tn, f_fn,
                                                                                                      m_tp, m_fp, m_tn,
                                                                                                      m_fn)

            test_loss_values.append(running_loss_test / len(test_loader))

            f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, aod, eod, spd = metrics(
                TP, FP, FN, f_tp, f_fp, f_fn, m_tp, m_fp, m_fn, TN, f_tn, m_tn)

            f1.append(f1_score_prediction)
            F_F1.append(f1_female)
            M_F1.append(f1_male)
            AOD.append(aod)
            SPD.append(spd)
            EOD.append(eod)

            # if epoch == 99:
            #     plt.plot(loss_values, label='Train Loss')
            #     plt.plot(test_loss_values, label='Test Loss')
            #     plt.xlabel('Epoch')
            #     plt.ylabel('Loss')
            #     title_loss = 'Loss over Epochs for LR Model on COMPAS - Client ' + str(c + 1)
            #     plt.title(title_loss)
            #     plt.legend(loc="upper right")
            #     plt.show()

            if c == 0:
                res = (
                pd  .DataFrame(columns = features_1, index = d_test_1.index)
                    .add_suffix('_partial')
                    .join(d_test_1)
                    .assign(prediction=predictions)
                    .assign(round=epoch)
                )
            else:
                res = (
                    pd.DataFrame(columns=features_2, index=d_test_2.index)
                        .add_suffix('_partial')
                        .join(d_test_2)
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

        if c == 0:
            all_acc_1.append(test_acc[len(test_acc) - 1])
            all_f_acc_1.append(F_ACC[len(F_ACC) - 1])
            all_m_acc_1.append(M_ACC[len(M_ACC) - 1])
            all_test_error_1.append(test_error[len(test_error) - 1])
            all_F_ERR_1.append(F_ERR[len(F_ERR) - 1])
            all_M_ERR_1.append(M_ERR[len(M_ERR) - 1])
            all_tp_1.append(tp[len(tp) - 1])
            all_tn_1.append(tn[len(tn) - 1])
            all_fp_1.append(fp[len(fp) - 1])
            all_fn_1.append(fn[len(fn) - 1])
            all_f1_1.append(f1[len(f1) - 1])
            all_F_TP_1.append(F_TP[len(F_TP) - 1])
            all_F_FP_1.append(F_FP[len(F_FP) - 1])
            all_F_TN_1.append(F_TN[len(F_TN) - 1])
            all_F_FN_1.append(F_FN[len(F_FN) - 1])
            all_F_F1_1.append(F_F1[len(F_F1) - 1])
            all_M_TP_1.append(M_TP[len(M_TP) - 1])
            all_M_FP_1.append(M_FP[len(M_FP) - 1])
            all_M_TN_1.append(M_TN[len(M_TN) - 1])
            all_M_FN_1.append(M_FN[len(M_FN) - 1])
            all_M_F1_1.append(M_F1[len(M_F1) - 1])
            all_EOD_1.append(EOD[len(EOD) - 1])
            all_SPD_1.append(SPD[len(SPD) - 1])
            all_AOD_1.append(AOD[len(AOD) - 1])
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            total_time = sum(times)
            all_times_1.append(total_time)
        elif c == 1:
            all_acc_2.append(test_acc[len(test_acc) - 1])
            all_f_acc_2.append(F_ACC[len(F_ACC) - 1])
            all_m_acc_2.append(M_ACC[len(M_ACC) - 1])
            all_test_error_2.append(test_error[len(test_error) - 1])
            all_F_ERR_2.append(F_ERR[len(F_ERR) - 1])
            all_M_ERR_2.append(M_ERR[len(M_ERR) - 1])
            all_tp_2.append(tp[len(tp) - 1])
            all_tn_2.append(tn[len(tn) - 1])
            all_fp_2.append(fp[len(fp) - 1])
            all_fn_2.append(fn[len(fn) - 1])
            all_f1_2.append(f1[len(f1) - 1])
            all_F_TP_2.append(F_TP[len(F_TP) - 1])
            all_F_FP_2.append(F_FP[len(F_FP) - 1])
            all_F_TN_2.append(F_TN[len(F_TN) - 1])
            all_F_FN_2.append(F_FN[len(F_FN) - 1])
            all_F_F1_2.append(F_F1[len(F_F1) - 1])
            all_M_TP_2.append(M_TP[len(M_TP) - 1])
            all_M_FP_2.append(M_FP[len(M_FP) - 1])
            all_M_TN_2.append(M_TN[len(M_TN) - 1])
            all_M_FN_2.append(M_FN[len(M_FN) - 1])
            all_M_F1_2.append(M_F1[len(M_F1) - 1])
            all_EOD_2.append(EOD[len(EOD) - 1])
            all_SPD_2.append(SPD[len(SPD) - 1])
            all_AOD_2.append(AOD[len(AOD) - 1])
            end = time.time()
            elapsed = end - start
            times.append(elapsed)
            total_time = sum(times)
            all_times_2.append(total_time)
        
        for col, encoder in encoders.items():
            results.loc[:, col] = encoder.inverse_transform(results[col])

        if c == 0:
            all_roc_1.append(plot_roc_curves(results, 'prediction', 'two_year_recid', size=(7, 5), fname='./results/roc.png'))
        elif c == 1:
            all_roc_2.append(plot_roc_curves(results, 'prediction', 'two_year_recid', size=(7, 5), fname='./results/roc.png'))

print('Client One')
print('*******************')
print('*       all       *')
print('*******************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_acc_1)))
print('Test Error: {0:1.3f}'.format(np.mean(all_test_error_1)))
print(
    'TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_tp_1), np.mean(all_fp_1), np.mean(all_tn_1),
                                                                      np.mean(all_fn_1)))
print('EOD: {0:1.4f}'.format(np.mean(all_EOD_1)))
print('SPD: {0:1.4f}'.format(np.mean(all_SPD_1)))
print('AOD: {0:1.4f}'.format(np.mean(all_AOD_1)))
print('F1: {0:1.3f}'.format(np.mean(all_f1_1)))
print("")
print('**********************')
print('*       Female       *')
print('**********************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_f_acc_1)))
print('Test Error: {0:1.3f}'.format(np.mean(all_F_ERR_1)))
print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_F_TP_1), np.mean(all_F_FP_1),
                                                                        np.mean(all_F_TN_1), np.mean(all_F_FN_1)))
print('F1: {0:1.3f}'.format(np.mean(all_F_F1_1)))
print("")
print('********************')
print('*       Male       *')
print('********************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_m_acc_1)))
print('Test Error: {0:1.3f}'.format(np.mean(all_M_ERR_1)))
print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_M_TP_1), np.mean(all_M_FP_1),
                                                                        np.mean(all_M_TN_1), np.mean(all_M_FN_1)))
print('F1: {0:1.3f}'.format(np.mean(all_M_F1_1)))

print('Train Time: {0:1.2f}'.format(np.mean(all_times_1)))
print('AUROC: {0:1.2f}'.format(np.mean(all_roc_1)))

print('Client Two')
print('*******************')
print('*       all       *')
print('*******************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_acc_2)))
print('Test Error: {0:1.3f}'.format(np.mean(all_test_error_2)))
print(
    'TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_tp_2), np.mean(all_fp_2), np.mean(all_tn_2),
                                                                      np.mean(all_fn_2)))
print('EOD: {0:1.4f}'.format(np.mean(all_EOD_2)))
print('SPD: {0:1.4f}'.format(np.mean(all_SPD_2)))
print('AOD: {0:1.4f}'.format(np.mean(all_AOD_2)))
print('F1: {0:1.3f}'.format(np.mean(all_f1_2)))
print("")
print('**********************')
print('*       Female       *')
print('**********************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_f_acc_2)))
print('Test Error: {0:1.3f}'.format(np.mean(all_F_ERR_2)))
print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_F_TP_2), np.mean(all_F_FP_2),
                                                                        np.mean(all_F_TN_2), np.mean(all_F_FN_2)))
print('F1: {0:1.3f}'.format(np.mean(all_F_F1_2)))
print("")
print('********************')
print('*       Male       *')
print('********************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_m_acc_2)))
print('Test Error: {0:1.3f}'.format(np.mean(all_M_ERR_2)))
print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_M_TP_2), np.mean(all_M_FP_2),
                                                                        np.mean(all_M_TN_2), np.mean(all_M_FN_2)))
print('F1: {0:1.3f}'.format(np.mean(all_M_F1_2)))

print('Train Time: {0:1.2f}'.format(np.mean(all_times_2)))
print('AUROC: {0:1.2f}'.format(np.mean(all_roc_2)))

