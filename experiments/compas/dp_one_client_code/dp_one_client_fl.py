import time
import json
import warnings
import torch
import torch.nn as nn
from fairtorch import DemographicParityLoss
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from utils import seed_everything, plot_roc_curves, get_data, confusion_matrix, metrics
from models import LR_combo, LR_HyperNet, NN_combo, NN_HyperNet
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

m = "dp-neural-net-fl"

no_cuda=False
gpus='3'
device = torch.cuda.set_device(3)

seed_everything(0)


data_train, data_test, features, d_train, d_test, encoders = get_data()
inter_results = []

learning_rate = [.1, .05, .01, .005, .003, .001, .0005, .0003, .0001, .00005, .00003, .00001]
alphas = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

all_acc, all_f_acc, all_m_acc = [], [], []
all_test_error, all_F_ERR, all_M_ERR = [], [], []
all_tp, all_tn, all_fp, all_fn, all_f1 = [], [], [], [], []
all_F_TP, all_F_FP, all_F_TN, all_F_FN, all_F_F1 = [], [], [], [], []
all_M_TP, all_M_FP, all_M_TN, all_M_FN, all_M_F1 = [], [], [], [], []
all_EOD, all_SPD, all_AOD = [], [], []
all_times, all_roc = [], []

for i, lr in enumerate(learning_rate):
    for j, alph in enumerate(alphas):
        for i in range(1):
            #print('Round: ', i)

            if m == "dp-log-reg-fl":
                model = LR_combo(input_size=9, vector_size=9)
                hnet = LR_HyperNet(vector_size=9, hidden_dim=9)
                l_o = 1e-5
                l_i = 5e-4
                step = 5
                ep = 25
                alpha = 50
                wd = 0
            elif m == "dp-neural-net-fl":
                model = NN_combo(input_size=9, vector_size=9)
                hnet = NN_HyperNet(vector_size=9, hidden_dim=9)
                l_o = lr
                l_i = 5e-4#1e-3
                step = 5
                ep = 50
                alpha = alph
                wd = 0.0000001

            model = model.double()
            hnet = hnet.double()

            hnet=hnet.to(device)
            model=model.to(device)

            optimizer = torch.optim.Adam(hnet.parameters(), lr= l_o, weight_decay=wd)
            inner_optimizer = torch.optim.Adam(model.parameters(), lr = l_i)
            loss = nn.BCELoss(reduction='mean')   #binary logarithmic loss function
            dp_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=alpha)

            train_loader = DataLoader(data_train, shuffle = True, batch_size = 128)
            test_loader = DataLoader(data_test, shuffle = False, batch_size= 128)

            loss_values =[]
            test_loss_values = []
            test_error, F_ERR, M_ERR = [], [], []

            acc_values, test_acc, F_ACC, M_ACC = [], [], [], []

            results, final_results = [],[]

            tp, tn, fp, fn, f1 = [], [], [], [], []
            F_TP, F_FP, F_TN, F_FN, F_F1 = [], [], [], [], []
            M_TP, M_FP, M_TN, M_FN, M_F1 = [], [], [], [], []
            EOD, SPD, AOD = [], [], []

            times = []

            # Train model
            steps = step
            epochs = ep
            place = 0
            start_epoch = time.time()
            for step in range(steps):
                hnet.train()
                torch.cuda.synchronize()
                start_epoch = time.time()

                average_dp = 0

                for epoch in range(epochs):
                    place = place + 1
                    model.train()
                    running_loss = 0.0
                    running_dp_loss = 0
                    correct = 0

                    for i, (x, y, s) in enumerate(train_loader):
                        # Generate weights on first round of the inner loop
                        if epoch == 0 and i == 0:
                            context_vectors, avg_context_vector, prediction_vector = model(x.to(device), context_only=True)

                            weights = hnet(avg_context_vector)
                            net_dict = model.state_dict()
                            hnet_dict = {k: v for k, v in weights.items() if k in net_dict}
                            net_dict.update(hnet_dict)
                            model.load_state_dict(net_dict)

                            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

                        inner_optimizer.zero_grad()
                        optimizer.zero_grad()

                        y_ = model(x.to(device), context_only=False)
                        demo_parity = dp_loss(x.float(), y_.float(), s.float())
                        err = loss(y_, y.unsqueeze(1).to(device)) + demo_parity
                        running_dp_loss += demo_parity
                        running_loss += err.item() * x.size(0)
                        err.backward()
                        inner_optimizer.step()

                        preds = y_.detach().cpu().round().reshape(1, len(y_))
                        correct += (preds.eq(y)).sum().item()

                    average_dp = running_dp_loss / len(train_loader)

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
                    total = 0.0
                    correct = 0.0

                    with torch.no_grad():
                        for i, (x, y, s) in enumerate(test_loader):
                            pred = model(x.to(device), context_only=False)
                            test_err = loss(pred.flatten(), y.to(device))
                            test_err = test_err.mean()
                            running_loss_test += test_err.item() * x.size(0)
                            preds = pred.detach().cpu().round().reshape(1, len(pred))
                            predictions.extend(preds.flatten().numpy())
                            correct += (preds.eq(y)).sum().item()

                            predicted_prediction = preds.type(torch.IntTensor).numpy().reshape(-1)
                            labels_pred = y.type(torch.IntTensor).numpy().reshape(-1)

                            TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn = confusion_matrix(s,
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

                    # if epoch == epochs - 1 and step == steps - 1:
                    #     plt.plot(loss_values, label='Train Loss')
                    #     plt.plot(test_loss_values, label='Test Loss')
                    #     plt.xlabel('Epoch')
                    #     plt.ylabel('Loss')
                    #     plt.title('Loss over Epochs for FL HN Adaptive LR Model on COMPAS')
                    #     plt.legend(loc="upper right")
                    #     plt.show()

                    res = (
                    pd  .DataFrame(columns = features, index = d_test.index)
                        .add_suffix('_partial')
                        .join(d_test)
                        .assign(prediction=predictions)
                        .assign(round=(step+1)*epoch)
                    )

                    results.append(res)
                    if epoch == epochs - 1:
                        final_results.append(res) #appended only on the last epoch
                    test_acc.append(accuracy) #appended each epoch
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

                all_results = pd.concat(results)
                final_final_results = pd.concat(final_results)

                optimizer.zero_grad()
                final_state = model.state_dict()

                # calculating delta theta
                #delta_theta = OrderedDict({k: inner_state[k] - final_state[k] + average_dp.cpu()  for k in weights.keys()})
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

                # print("With dp:" , delta_theta, "\n")
                # print("WIthout dp:", delta_theta_no_add, "\n")
                # print("Average:", average_dp.cpu())
                # exit(1)
                # calculating phi gradient
                hnet_grads = torch.autograd.grad(
                    list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
                )

                # update hnet weights
                for p, g in zip(hnet.parameters(), hnet_grads):
                    p.grad = g

                # torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
                optimizer.step()

            # average_test_acc = sum(test_acc) / len(test_acc)
            # female_test_acc = sum(F_ACC) / len(F_ACC)
            # male_test_acc = sum(M_ACC) / len(M_ACC)

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
            all_times.append(sum(times))

            for col, encoder in encoders.items():
                all_results.loc[:, col] = encoder.inverse_transform(all_results[col])

            all_roc.append(
                plot_roc_curves(final_final_results, 'prediction', 'two_year_recid', size=(7, 5), fname='./results/roc.png'))

        add_to_dict = {'lr': lr, 'alpha': alph, 'avg_acc': np.mean(all_acc), 'EOD': np.mean(all_EOD), 'AOD': np.mean(all_AOD), 'SPD': np.mean(all_SPD)}
        inter_results.append(add_to_dict)
        print(add_to_dict)
        #print(all_acc)
        # print('*******************')
        # print('*       all       *')
        # print('*******************')
        # print('Test Accuracy: {0:1.3f}'.format(np.mean(all_acc)))
        # print('Test Error: {0:1.3f}'.format(np.mean(all_test_error)))
        # print(
        #     'TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_tp), np.mean(all_fp), np.mean(all_tn),
        #                                                                       np.mean(all_fn)))
        # print('EOD: {0:1.4f}'.format(np.mean(all_EOD)))
        # print('SPD: {0:1.4f}'.format(np.mean(all_SPD)))
        # print('AOD: {0:1.4f}'.format(np.mean(all_AOD)))
        # print('F1: {0:1.3f}'.format(np.mean(all_f1)))
        # print("")
        # print('**********************')
        # print('*       Female       *')
        # print('**********************')
        # print('Test Accuracy: {0:1.3f}'.format(np.mean(all_f_acc)))
        # print('Test Error: {0:1.3f}'.format(np.mean(all_F_ERR)))
        # print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_F_TP), np.mean(all_F_FP),
        #                                                                         np.mean(all_F_TN), np.mean(all_F_FN)))
        # print('F1: {0:1.3f}'.format(np.mean(all_F_F1)))
        # print("")
        # print('********************')
        # print('*       Male       *')
        # print('********************')
        # print('Test Accuracy: {0:1.3f}'.format(np.mean(all_m_acc)))
        # print('Test Error: {0:1.3f}'.format(np.mean(all_M_ERR)))
        # print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(np.mean(all_M_TP), np.mean(all_M_FP),
        #                                                                         np.mean(all_M_TN), np.mean(all_M_FN)))
        # print('F1: {0:1.3f}'.format(np.mean(all_M_F1)))
        #
        # print('Train Time: {0:1.2f}'.format(np.mean(all_times)))
        # print('AUROC: {0:1.2f}'.format(np.mean(all_roc)))

#textfile = open("all_results_dp.txt", "w")
# for element in all_results:
#     textfile.write(inter_results)
# textfile.close()

with open("all_results_dp.txt", "w") as f:
    for item in inter_results:
        f.write("%s\n" % item)
