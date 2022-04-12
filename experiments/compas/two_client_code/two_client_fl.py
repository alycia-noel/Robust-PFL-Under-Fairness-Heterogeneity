import time
import warnings
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import OrderedDict
from utils import seed_everything, plot_roc_curves, get_data, confusion_matrix, metrics
from models import LR_combo, NN_combo, LR_HyperNet, NN_HyperNet

warnings.filterwarnings("ignore")

no_cuda=False
gpus='3'
device = torch.cuda.set_device(3)

seed_everything(0)

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
m = "log-reg-two-fl"
for i in range(1):
    print('Round:', i)
    seed_everything(0)
    data_train_1, data_test_1, data_train_2, data_test_2, features_1, features_2, decision_1, decision_2, d_test_1, d_test_2, encoders = get_data()

    if m == "log-reg-two-fl":
        c1_model = LR_combo(input_size=10, vector_size=10, hidden_size=100)
        c2_model = LR_combo(input_size=10, vector_size=10, hidden_size=100)
        hnet = LR_HyperNet(vector_size=10, hidden_dim=100, num_hidden=3) #2
        c1_l = .003 #.005
        c2_l = .003
        o_l = .003 #.002
        step = 5
        ep = 20
        wd = 0
    elif m == "neural-net-two-fl":
        c1_model = NN_combo(input_size=10, vector_size=10, hidden_size=100)
        c2_model = NN_combo(input_size=10, vector_size=10, hidden_size=100)
        hnet = NN_HyperNet(vector_size=10, hidden_dim=100, num_hidden = 4)
        c1_l = .007
        c2_l = .002
        o_l = .0005
        step = 5
        ep = 50
        wd= 0.0000001

    c1_model = c1_model.double()
    c2_model = c2_model.double()
    hnet = hnet.double()

    no_cuda=False
    gpus='4'
    device = torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")

    hnet=hnet.to(device)
    c1_model=c1_model.to(device)
    c2_model= c2_model.to(device)

    optimizer = torch.optim.Adam(hnet.parameters(), lr= o_l)
    c1_inner_optimizer = torch.optim.Adam(c1_model.parameters(), lr = c1_l, weight_decay=wd)
    c2_inner_optimizer = torch.optim.Adam(c2_model.parameters(), lr = c2_l, weight_decay=wd)

    loss = nn.BCELoss(reduction='mean')   #binary logarithmic loss function

    c1_loss_values, c1_test_loss_values =[], []
    c1_acc_values, c1_test_acc, c1_f_acc, c1_m_acc = [], [], [], []
    c1_test_error, c1_f_err, c1_m_err = [], [], []
    c1_tp, c1_tn, c1_fp, c1_fn = [], [], [], []
    c1_f_tp, c1_f_tn, c1_f_fp, c1_f_fn = [], [], [], []
    c1_m_tp, c1_m_tn, c1_m_fp, c1_m_fn = [], [], [], []
    c1_times = []
    c1_f1, c1_f_f1, c1_m_f1 = [], [], []
    c1_EOD, c1_SPD, c1_AOD = [], [], []
    c1_results, c1_final_results, c1_final_final_results, c1_all_results = [], [], [], []

    c2_loss_values, c2_test_loss_values =[], []
    c2_acc_values, c2_test_acc, c2_f_acc, c2_m_acc = [], [], [], []
    c2_test_error, c2_f_err, c2_m_err = [], [], []
    c2_tp, c2_tn, c2_fp, c2_fn = [], [], [], []
    c2_f_tp, c2_f_tn, c2_f_fp, c2_f_fn = [], [], [], []
    c2_m_tp, c2_m_tn, c2_m_fp, c2_m_fn = [], [], [], []
    c2_times = []
    c2_f1, c2_f_f1, c2_m_f1 = [], [], []
    c2_EOD, c2_SPD, c2_AOD = [], [], []
    c2_results, c2_final_results, c2_final_final_results, c2_all_results = [], [], [], []

    delta_theta_1 = OrderedDict()
    delta_theta_2 = OrderedDict()

    hnet_grads_1 = ()
    hnet_grads_2 = ()

    # Train model
    steps = step
    epochs = ep

    torch.cuda.synchronize()
    for step in range(steps):
        #Need to save model so client both have the same starting point
        torch.save(hnet.state_dict(), 'hnet_baseline_weights.pth')

        for c in range(clients):
            if c == 0:
                data_train = data_train_1
                data_test = data_test_1
            else:
                data_train = data_train_2
                data_test = data_test_2


            train_loader = DataLoader(data_train, shuffle=True, batch_size=256)
            test_loader = DataLoader(data_test, shuffle=False, batch_size=256)

            #load baseline
            hnet.load_state_dict(torch.load('hnet_baseline_weights.pth'))

            hnet.train()
            start_epoch = time.time()

            for epoch in range(epochs):
                if c == 0:
                    model = c1_model
                    inner_optimizer = c1_inner_optimizer
                if c == 1:
                    model = c2_model
                    inner_optimizer = c2_inner_optimizer

                model.train()
                running_loss = 0.0
                correct = 0

                for i, (x, y) in enumerate(train_loader):
                    if epoch == 0 and i == 0:
                        context_vectors, avg_context_vector, prediction_vector = model(x.to(device), context_only=True)

                        weights = hnet(avg_context_vector, torch.tensor([c], dtype=torch.long).to(device))
                        net_dict = model.state_dict()
                        hnet_dict = {k: v for k, v in weights.items() if k in net_dict}
                        net_dict.update(hnet_dict)
                        model.load_state_dict(net_dict)

                        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

                    inner_optimizer.zero_grad()
                    optimizer.zero_grad()

                    y_ = model(x.to(device), context_only=False)
                    err = loss(y_, y.unsqueeze(1).to(device))
                    err = err.mean()

                    err.backward()
                    running_loss += err.item() * x.size(0)
                    torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
                    inner_optimizer.step()

                    preds = y_.detach().cpu().round().reshape(1, len(y_))
                    correct += (preds.eq(y)).sum().item()

                accuracy = (100 * correct / len(data_train))

                if c == 0:
                    c1_acc_values.append(accuracy)
                    c1_loss_values.append(running_loss / len(train_loader))
                else:
                    c2_acc_values.append(accuracy)
                    c2_loss_values.append(running_loss / len(train_loader))

                #print('Step: {5}/{6};\tEpoch: {0}/{1};\tLoss: {2:1.3f};\tAcc: {3:1.3f};\tPercent: {4:1.2f}%'.format(epoch+1, epochs, running_loss / len(train_loader), accuracy,(100*place)/(epochs*steps), step+1, steps))

                # Eval Model
                model.eval()
                predictions = []
                running_loss_test = 0
                TP, FP, FN, TN = 0, 0, 0, 0
                f_tp, f_fp, f_tn, f_fn = 0, 0, 0, 0
                m_tp, m_fp, m_tn, m_fn = 0, 0, 0, 0
                total = 0.0
                correct = 0.0
                with torch.no_grad():
                    for i, (x, y) in enumerate(test_loader):
                        pred = model(x.to(device), context_only=False)
                        test_err = loss(pred.flatten(), y.to(device))
                        test_err = test_err.mean()
                        running_loss_test += test_err.item() * x.size(0)

                        preds = pred.detach().cpu().round().reshape(1, len(pred))
                        predictions.extend(preds.flatten().numpy())
                        correct += (preds.eq(y)).sum().item()

                        predicted_prediction = preds.type(torch.IntTensor).numpy().reshape(-1)
                        labels_pred = y.type(torch.IntTensor).numpy().reshape(-1)

                        TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn = confusion_matrix(x,
                                                                                                          predicted_prediction,
                                                                                                          labels_pred,
                                                                                                          TP,
                                                                                                          FP, FN, TN,
                                                                                                          f_tp,
                                                                                                          f_fp, f_tn,
                                                                                                          f_fn,
                                                                                                          m_tp, m_fp,
                                                                                                          m_tn,
                                                                                                          m_fn)

                if c == 0:
                    c1_test_loss_values.append(running_loss_test / len(test_loader))
                else:
                    c2_test_loss_values.append(running_loss_test / len(test_loader))

                f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, aod, eod, spd = metrics(
                    TP, FP, FN, f_tp, f_fp, f_fn, m_tp, m_fp, m_fn, TN, f_tn, m_tn)

                if c == 0:
                    c1_f1.append(f1_score_prediction)
                    c1_f_f1.append(f1_female)
                    c1_m_f1.append(f1_male)
                    loss_values = c1_loss_values
                    test_loss_values = c1_test_loss_values
                else:
                    c2_f1.append(f1_score_prediction)
                    c2_f_f1.append(f1_female)
                    c2_m_f1.append(f1_male)
                    loss_values = c2_loss_values
                    test_loss_values = c2_test_loss_values

                # if epoch == epochs - 1 and step == steps-1:
                #     plt.plot(loss_values, label='Train Loss')
                #     plt.plot(test_loss_values, label='Test Loss')
                #     plt.xlabel('Epoch')
                #     plt.ylabel('Loss')
                #     title_loss = 'Loss over Epochs for FL HN Adaptive LR Model on COMPAS - Client ' + str(c + 1)
                #     plt.title(title_loss)
                #     plt.legend(loc="upper right")
                #     plt.show()

                if c == 0:
                    res = (
                        pd.DataFrame(columns=features_1, index=d_test_1.index)
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

                if c == 0:
                    c1_results.append(res)
                    if epoch == epochs - 1:
                        c1_final_results.append(res)

                    c1_test_acc.append(accuracy)
                    c1_f_acc.append(f_acc)
                    c1_m_acc.append(m_acc)
                    c1_test_error.append(error)
                    c1_f_err.append(f_err)
                    c1_m_err.append(m_err)

                    c1_tn.append(TN)
                    c1_tp.append(TP)
                    c1_fn.append(FN)
                    c1_fp.append(FP)

                    c1_f_tn.append(f_tn)
                    c1_f_tp.append(f_tp)
                    c1_f_fn.append(f_fn)
                    c1_f_fp.append(f_fp)

                    c1_m_tn.append(m_tn)
                    c1_m_tp.append(m_tp)
                    c1_m_fn.append(m_fn)
                    c1_m_fp.append(m_fp)

                    c1_EOD.append(eod)
                    c1_SPD.append(spd)
                    c1_AOD.append(aod)

                else:
                    c2_results.append(res)
                    if epoch == epochs - 1:
                        c2_final_results.append(res)

                    c2_test_acc.append(accuracy)
                    c2_f_acc.append(f_acc)
                    c2_m_acc.append(m_acc)
                    c2_test_error.append(error)
                    c2_f_err.append(f_err)
                    c2_m_err.append(m_err)

                    c2_tn.append(TN)
                    c2_tp.append(TP)
                    c2_fn.append(FN)
                    c2_fp.append(FP)

                    c2_f_tn.append(f_tn)
                    c2_f_tp.append(f_tp)
                    c2_f_fn.append(f_fn)
                    c2_f_fp.append(f_fp)

                    c2_m_tn.append(m_tn)
                    c2_m_tp.append(m_tp)
                    c2_m_fn.append(m_fn)
                    c2_m_fp.append(m_fp)

                    c2_EOD.append(eod)
                    c2_SPD.append(spd)
                    c2_AOD.append(aod)

            if c == 0:
                c1_all_results = pd.concat(c1_results)
                c1_final_final_results = pd.concat(c1_final_results)
            elif c == 1:
                c2_all_results = pd.concat(c2_results)
                c2_final_final_results = pd.concat(c2_final_results)

            optimizer.zero_grad()
            final_state = model.state_dict()

            # calculating delta theta
            if c == 0:
                delta_theta_1 = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
                hnet_grads_1 = torch.autograd.grad(
                    list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta_1.values())
                )
            elif c == 1:
                delta_theta_2 = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
                hnet_grads_2 = torch.autograd.grad(
                    list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta_2.values())
                )

        hnet.load_state_dict(torch.load('hnet_baseline_weights.pth'))

        # update hnet weights from client 1
        for p, g in zip(hnet.parameters(), hnet_grads_1):
            p.grad = g

        # update hnet weights from client 2
        for p, g in zip(hnet.parameters(), hnet_grads_2):
            p.grad = g

        # torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

    all_acc_1.append(c1_test_acc[len(c1_test_acc) - 1])
    all_f_acc_1.append(c1_f_acc[len(c1_f_acc) - 1])
    all_m_acc_1.append(c1_m_acc[len(c1_m_acc) - 1])
    all_test_error_1.append(c1_test_error[len(c1_test_error) - 1])
    all_F_ERR_1.append(c1_f_err[len(c1_f_err) - 1])
    all_M_ERR_1.append(c1_m_err[len(c1_m_err) - 1])
    all_tp_1.append(c1_tp[len(c1_tp) - 1])
    all_tn_1.append(c1_tn[len(c1_tn) - 1])
    all_fp_1.append(c1_fp[len(c1_fp) - 1])
    all_fn_1.append(c1_fn[len(c1_fn) - 1])
    all_f1_1.append(c1_f1[len(c1_f1) - 1])
    all_F_TP_1.append(c1_f_tp[len(c1_f_tp) - 1])
    all_F_FP_1.append(c1_f_fp[len(c1_f_fp) - 1])
    all_F_TN_1.append(c1_f_tn[len(c1_f_tn) - 1])
    all_F_FN_1.append(c1_f_fn[len(c1_f_fn) - 1])
    all_F_F1_1.append(c1_f1[len(c1_f1) - 1])
    all_M_TP_1.append(c1_m_tp[len(c1_m_tp) - 1])
    all_M_FP_1.append(c1_m_fp[len(c1_m_fp) - 1])
    all_M_TN_1.append(c1_m_tn[len(c1_m_tn) - 1])
    all_M_FN_1.append(c1_m_fn[len(c1_m_fn) - 1])
    all_M_F1_1.append(c1_m_f1[len(c1_m_f1) - 1])
    all_EOD_1.append(c1_EOD[len(c1_EOD) - 1])
    all_SPD_1.append(c1_SPD[len(c1_SPD) - 1])
    all_AOD_1.append(c1_AOD[len(c1_AOD) - 1])
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    c1_times.append(elapsed)
    all_times_1.append(sum(c1_times))

    for col, encoder in encoders.items():
        c1_all_results.loc[:, col] = encoder.inverse_transform(c1_all_results[col])

    all_roc_1.append(plot_roc_curves(c1_final_final_results, 'prediction', 'two_year_recid', size=(7, 5),
                                     fname='./results/roc.png'))

    all_acc_2.append(c2_test_acc[len(c2_test_acc) - 1])
    all_f_acc_2.append(c2_f_acc[len(c2_f_acc) - 1])
    all_m_acc_2.append(c2_m_acc[len(c2_m_acc) - 1])
    all_test_error_2.append(c2_test_error[len(c2_test_error) - 1])
    all_F_ERR_2.append(c2_f_err[len(c2_f_err) - 1])
    all_M_ERR_2.append(c2_m_err[len(c2_m_err) - 1])
    all_tp_2.append(c2_tp[len(c2_tp) - 1])
    all_tn_2.append(c2_tn[len(c2_tn) - 1])
    all_fp_2.append(c2_fp[len(c2_fp) - 1])
    all_fn_2.append(c2_fn[len(c2_fn) - 1])
    all_f1_2.append(c2_f1[len(c2_f1) - 1])
    all_F_TP_2.append(c2_f_tp[len(c2_f_tp) - 1])
    all_F_FP_2.append(c2_f_fp[len(c2_f_fp) - 1])
    all_F_TN_2.append(c2_f_tn[len(c2_f_tn) - 1])
    all_F_FN_2.append(c2_f_fn[len(c2_f_fn) - 1])
    all_F_F1_2.append(c2_f1[len(c2_f1) - 1])
    all_M_TP_2.append(c2_m_tp[len(c2_m_tp) - 1])
    all_M_FP_2.append(c2_m_fp[len(c2_m_fp) - 1])
    all_M_TN_2.append(c2_m_tn[len(c2_m_tn) - 1])
    all_M_FN_2.append(c2_m_fn[len(c2_m_fn) - 1])
    all_M_F1_2.append(c2_m_f1[len(c2_m_f1) - 1])
    all_EOD_2.append(c2_EOD[len(c2_EOD) - 1])
    all_SPD_2.append(c2_SPD[len(c2_SPD) - 1])
    all_AOD_2.append(c2_AOD[len(c2_AOD) - 1])
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    c2_times.append(elapsed)
    all_times_2.append(sum(c2_times))
    for col, encoder in encoders.items():
        c2_all_results.loc[:, col] = encoder.inverse_transform(c2_all_results[col])

    all_roc_2.append(plot_roc_curves(c2_final_final_results, 'prediction', 'two_year_recid', size=(7, 5),
                                     fname='./results/roc.png'))
print('Client One')
print('*******************')
print('*       all       *')
print('*******************')
print('Test Accuracy: {0:1.3f}'.format(np.mean(all_acc_1)))
print('Test Error: {0:1.3f}'.format(np.mean(all_test_error_1)))
print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_tp_1), np.mean(all_fp_1), np.mean(all_tn_1),
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

# add_to_dict = {'num_hidden': num,'o_l': o, 'c1_l': c1, 'c2_l': c2, 'c1 acc': np.mean(all_acc_1), 'c2 acc':np.mean(all_acc_2) }
# hyperparam.append(add_to_dict)
# #
# with open('hyperparam.txt', 'w') as f:
#     for item in hyperparam:
#         f.write("%s\n" % item)
#     f.close()