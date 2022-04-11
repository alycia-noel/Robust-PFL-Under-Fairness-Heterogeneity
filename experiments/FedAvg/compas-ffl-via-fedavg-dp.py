import time
import warnings
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import seed_everything, plot_roc_curves, get_data_dp, confusion_matrix_dp, metrics
from models import LR, NN
from fairtorch import DemographicParityLoss
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

no_cuda = False
gpus = '5'
device = torch.cuda.set_device(5)

def train_dp(model, train_loader, loss, optimizer, alpha):
    model.train()
    model = model.to(device)
    correct = 0
    running_loss = 0.0

    dp_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=alpha)

    for i, (x, y, s) in enumerate(train_loader):
        optimizer.zero_grad()
        y_, y_raw = model(x.to(device))
        err = loss(y_raw.flatten(), y.to(device)) + dp_loss(x.float(), y_raw.float(), s.float()).cpu()
        err = err.mean()
        running_loss += err.item() * x.size(0)
        err.backward()
        optimizer.step()

        preds = y_.round().reshape(1, len(y_))
        correct += (preds.eq(y)).sum().item()

    return (running_loss / len(train_loader)), (100 * correct / len(train_loader.dataset))


def validation_dp(model, test_loader, loss):
    model.eval()
    model = model.to(device)
    predictions = []
    TP, FP, FN, TN = 0, 0, 0, 0
    f_tp, f_fp, f_tn, f_fn = 0, 0, 0, 0
    m_tp, m_fp, m_tn, m_fn = 0, 0, 0, 0
    running_loss_test = 0.0
    correct = 0

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

            TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn = confusion_matrix_dp(s,
                                                                                                 predicted_prediction,
                                                                                                 labels_pred, TP, FP,
                                                                                                 FN, TN, f_tp, f_fp,
                                                                                                 f_tn, f_fn, m_tp, m_fp,
                                                                                                 m_tn, m_fn)

    running_loss_test /= len(test_loader)
    f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, aod, eod, spd = metrics(TP,
                                                                                                                  FP,
                                                                                                                  FN,
                                                                                                                  f_tp,
                                                                                                                  f_fp,
                                                                                                                  f_fn,
                                                                                                                  m_tp,
                                                                                                                  m_fp,
                                                                                                                  m_fn,
                                                                                                                  TN,
                                                                                                                  f_tn,
                                                                                                                  m_tn)

    return predictions, running_loss_test, f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, aod, eod, spd, TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn


def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples):
    with torch.no_grad():
        for i in range(number_of_samples):
            model_dict[name_of_models[i]].fc1.weight.data = main_model.fc1.weight.data.clone()
            model_dict[name_of_models[i]].fc1.bias.data = main_model.fc1.bias.data.clone()
            #
            # model_dict[name_of_models[i]].fc2.weight.data = main_model.fc2.weight.data.clone()
            # model_dict[name_of_models[i]].fc2.bias.data = main_model.fc2.bias.data.clone()
            #
            # model_dict[name_of_models[i]].fc3.weight.data = main_model.fc3.weight.data.clone()
            # model_dict[name_of_models[i]].fc3.bias.data = main_model.fc3.bias.data.clone()
            #
            # model_dict[name_of_models[i]].fc4.weight.data = main_model.fc4.weight.data.clone()
            # model_dict[name_of_models[i]].fc4.bias.data = main_model.fc4.bias.data.clone()
            #
            # model_dict[name_of_models[i]].fc5.weight.data = main_model.fc5.weight.data.clone()
            # model_dict[name_of_models[i]].fc5.bias.data = main_model.fc5.bias.data.clone()

        return model_dict


def create_model_optimizer_criterion_dict(number_of_samples):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)
        if m == "log-reg":
            model_info = LR(input_size=9)
        elif m == "neural-net":
            model_info = NN(input_size=9)

        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.Adam(model_info.parameters(), lr=learning_rate, weight_decay=wd)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.BCEWithLogitsLoss(reduction='mean')
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def start_train_end_node_process_print_some(number_of_samples, print_amount, alpha):
    running_loss_1 = []
    running_loss_2 = []

    for i in range(number_of_samples):

        results = []

        test_error, F_ERR, M_ERR = [], [], []

        acc_values, test_acc, F_ACC, M_ACC = [], [], [], []

        tp, tn, fp, fn, f1 = [], [], [], [], []
        F_TP, F_FP, F_TN, F_FN, F_F1 = [], [], [], [], []
        M_TP, M_FP, M_TN, M_FN, M_F1 = [], [], [], [], []
        EOD, SPD, AOD = [], [], []
        times = []

        if i == 0:
            data_train = data_train_1
            data_test = data_test_1
        if i == 1:
            data_train = data_train_2
            data_test = data_test_2

        train_dl = DataLoader(data_train, shuffle=True, batch_size=256)
        test_dl = DataLoader(data_test, shuffle=False, batch_size=256)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        start = time.time()
        for epoch in range(num_epoch):
            train_loss, train_accuracy = train_dp(model.double(), train_dl, criterion, optimizer, alpha)
            predictions, running_loss_test, f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, aod, eod, spd, TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn = validation_dp(
                model.double(), test_dl, criterion)


            f1.append(f1_score_prediction)
            F_F1.append(f1_female)
            M_F1.append(f1_male)
            AOD.append(aod)
            SPD.append(spd)
            EOD.append(eod)

            if i == 0:
                running_loss_1.append(running_loss_test)
                res = (
                    pd.DataFrame(columns=features_1, index=d_test_1.index)
                        .add_suffix('_partial')
                        .join(d_test_1)
                        .assign(prediction=predictions)
                        .assign(round=epoch)
                )
            else:
                running_loss_2.append(running_loss_test)
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

        #print("client: {}".format(i+1) + " | epoch: {:3.0f}".format(epoch + 1) + " | train accuracy: {:7.5f}".format(
        #    train_accuracy) + " | test accuracy: {:7.5f}".format(accuracy))

        results = pd.concat(results)

        if i == 0:
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
        elif i == 1:
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

        if i == 0:
            all_roc_1.append(
                plot_roc_curves(results, 'prediction', 'two_year_recid', size=(7, 5), fname='./results/roc.png'))
        elif i == 1:
            all_roc_2.append(
                plot_roc_curves(results, 'prediction', 'two_year_recid', size=(7, 5), fname='./results/roc.png'))

    return running_loss_1, running_loss_2

def get_averaged_weights(model_dict, number_of_samples):
    fc1_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc1.weight.shape)
    fc1_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc1.bias.shape)

    # fc2_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc2.weight.shape)
    # fc2_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc2.bias.shape)
    #
    # fc3_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc3.weight.shape)
    # fc3_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc3.bias.shape)
    #
    # fc4_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc4.weight.shape)
    # fc4_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc4.bias.shape)
    #
    # fc5_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc5.weight.shape)
    # fc5_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc5.bias.shape)

    with torch.no_grad():
        for i in range(number_of_samples):
            fc1_mean_weight += model_dict[name_of_models[i]].fc1.weight.data.clone()
            fc1_mean_bias += model_dict[name_of_models[i]].fc1.bias.data.clone()

            # fc2_mean_weight += model_dict[name_of_models[i]].fc2.weight.data.clone()
            # fc2_mean_bias += model_dict[name_of_models[i]].fc2.bias.data.clone()
            #
            # fc3_mean_weight += model_dict[name_of_models[i]].fc3.weight.data.clone()
            # fc3_mean_bias += model_dict[name_of_models[i]].fc3.bias.data.clone()
            #
            # fc4_mean_weight += model_dict[name_of_models[i]].fc4.weight.data.clone()
            # fc4_mean_bias += model_dict[name_of_models[i]].fc4.bias.data.clone()
            #
            # fc5_mean_weight += model_dict[name_of_models[i]].fc5.weight.data.clone()
            # fc5_mean_bias += model_dict[name_of_models[i]].fc5.bias.data.clone()

        fc1_mean_weight = fc1_mean_weight / number_of_samples
        fc1_mean_bias = fc1_mean_bias / number_of_samples

        # fc2_mean_weight = fc2_mean_weight / number_of_samples
        # fc2_mean_bias = fc2_mean_bias / number_of_samples
        #
        # fc3_mean_weight = fc3_mean_weight / number_of_samples
        # fc3_mean_bias = fc3_mean_bias / number_of_samples
        #
        # fc4_mean_weight = fc4_mean_weight / number_of_samples
        # fc4_mean_bias = fc4_mean_bias / number_of_samples
        #
        # fc5_mean_weight = fc5_mean_weight / number_of_samples
        # fc5_mean_bias = fc5_mean_bias / number_of_samples

    return fc1_mean_weight, fc1_mean_bias#, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias, fc4_mean_weight, fc4_mean_bias, fc5_mean_weight, fc5_mean_bias


def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model,model_dict, number_of_samples):
    # fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias, fc4_mean_weight, fc4_mean_bias, fc5_mean_weight, fc5_mean_bias = get_averaged_weights(model_dict, number_of_samples=number_of_samples)
    fc1_mean_weight, fc1_mean_bias = get_averaged_weights(model_dict, number_of_samples=number_of_samples)

    with torch.no_grad():
        main_model.fc1.weight.data = fc1_mean_weight.data.clone()
        main_model.fc1.bias.data = fc1_mean_bias.data.clone()

        # main_model.fc2.weight.data = fc2_mean_weight.data.clone()
        # main_model.fc2.bias.data = fc2_mean_bias.data.clone()
        #
        # main_model.fc3.weight.data = fc3_mean_weight.data.clone()
        # main_model.fc3.bias.data = fc3_mean_bias.data.clone()
        #
        # main_model.fc4.weight.data = fc4_mean_weight.data.clone()
        # main_model.fc4.bias.data = fc4_mean_bias.data.clone()
        #
        # main_model.fc5.weight.data = fc5_mean_weight.data.clone()
        # main_model.fc5.bias.data = fc5_mean_bias.data.clone()

    return main_model



seed_everything(0)
data_test_all, data_train_1, data_test_1, data_train_2, data_test_2, features_1, features_2, features_all, decision_1, decision_2, d_test_1, d_test_2, d_test_all, encoders = get_data_dp()
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

results_all = []
acc_all, F_ACC_all, M_ACC_all = [], [], []
test_error_all, F_ERR_all, M_ERR_all = [], [], []
tp_all, tn_all, fp_all, fn_all, f1_all = [], [], [], [], []
F_TP_all, F_FP_all, F_TN_all, F_FN_all, F_F1_all = [], [], [], [], []
M_TP_all, M_FP_all, M_TN_all, M_FN_all, M_F1_all = [], [], [], [], []
EOD_all, SPD_all, AOD_all = [], [], []
times_all, roc_all = [], []

number_of_samples = 2  # i.e. number of clients
batch_size = 128
print_amount = 0
m = "log-reg"
if m == "log-reg":
    main_model = LR(input_size=9)
    learning_rate = 7e-3
    learning_rate_global = 5e-4
    num_epoch = 10  # local
    wd = 0
if m == "neural-net":
    main_model = NN(input_size=9)
    learning_rate = 1e-3
    num_epoch = 25  # local
    learning_rate_global = 1e-3
    wd = 0.0000001

main_optimizer = torch.optim.Adam(main_model.parameters(), lr=learning_rate_global, weight_decay=wd)
loss = nn.BCEWithLogitsLoss(reduction='mean')
main_criterion = nn.BCEWithLogitsLoss()

model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(number_of_samples)

name_of_models = list(model_dict.keys())
name_of_optimizers = list(optimizer_dict.keys())
name_of_criterions = list(criterion_dict.keys())

test_dl = DataLoader(data_test_all, shuffle=False, batch_size=256)

print('~' * 22)
print('~~~ Start Training ~~~')
print('~' * 22, '\n')
results_main_model = []

alpha_results_1 = []
alpha_results_2 = []

alphas = [1, 5, 10, 25, 50, 75, 100]

for j, alph in enumerate(alphas):
    l_1, l_2 = [], []
    for i in range(10):
        model_dict = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples)
        loss_1, loss_2 = start_train_end_node_process_print_some(number_of_samples, print_amount, alpha=alph)

        start = time.time()
        main_model = set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict,
                                                                                      number_of_samples)
        end = time.time()
        predictions, running_loss_test, f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, aod, eod, spd, TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn = validation_dp(
            main_model.double(), test_dl, main_criterion)

        l_1.extend(loss_1)
        l_2.extend(loss_2)
        times_all.append(end - start)
        f1_all.append(f1_score_prediction)
        F_F1_all.append(f1_female)
        M_F1_all.append(f1_male)
        AOD_all.append(aod)
        SPD_all.append(spd)
        EOD_all.append(eod)

        res = (
            pd.DataFrame(columns=features_all, index=d_test_all.index)
                .add_suffix('_partial')
                .join(d_test_all)
                .assign(prediction=predictions)
                .assign(round=i)
        )

        results_main_model.append(res)
        acc_all.append(accuracy)
        F_ACC_all.append(f_acc)
        M_ACC_all.append(m_acc)
        test_error_all.append(error)
        F_ERR_all.append(f_err)
        M_ERR_all.append(m_err)

        tn_all.append(TN)
        tp_all.append(TP)
        fn_all.append(FN)
        fp_all.append(FP)

        F_TN_all.append(f_tn)
        F_TP_all.append(f_tp)
        F_FN_all.append(f_fn)
        F_FP_all.append(f_fp)

        M_TN_all.append(m_tn)
        M_TP_all.append(m_tp)
        M_FN_all.append(m_fn)
        M_FP_all.append(m_fp)

        # print("\nIteration", str(i + 1), ": main_model accuracy on all test data: {:7.4f}".format(accuracy))
        # print("\n")

    alpha_results_1.append(l_1)
    alpha_results_2.append(l_2)

plt.plot(alpha_results_1[0], label='a = 1')
plt.plot(alpha_results_1[1], label='a = 5')
plt.plot(alpha_results_1[2], label='a = 10')
plt.plot(alpha_results_1[3], label='a = 25')
plt.plot(alpha_results_1[4], label='a = 50')
plt.plot(alpha_results_1[5], label='a = 75')
plt.plot(alpha_results_1[6], label='a = 100')
plt.xlabel('Epoch')
plt.ylabel('Loss')
title_loss = 'Loss per Epochs for FFL via FedAvg using Fairness Weight a - Client ' + str(1)
plt.title(title_loss)
plt.legend(loc="upper right")
plt.show()

plt.plot(alpha_results_2[0], label='a = 1')
plt.plot(alpha_results_2[1], label='a = 5')
plt.plot(alpha_results_2[2], label='a = 10')
plt.plot(alpha_results_2[3], label='a = 25')
plt.plot(alpha_results_2[4], label='a = 50')
plt.plot(alpha_results_2[5], label='a = 75')
plt.plot(alpha_results_2[6], label='a = 100')
plt.xlabel('Epoch')
plt.ylabel('Loss')
title_loss = 'Loss per Epochs for FFL via FedAvg using Fairness Weight a - Client ' + str(2)
plt.title(title_loss)
plt.legend(loc="upper right")
plt.show()

# results_all = pd.concat(results_main_model)
#
# for col, encoder in encoders.items():
#     results_all.loc[:, col] = encoder.inverse_transform(results_all[col])
#
# roc_all.append(plot_roc_curves(results_all, 'prediction', 'two_year_recid', size=(7, 5), fname='./results/roc.png'))
#
# print('\n')
# print('~' * 30)
# print('~         Client One         ~')
# print('~' * 30)
# print('*******************')
# print('*       all       *')
# print('*******************')
# print('Test Accuracy: {0:1.3f}'.format(all_acc_1[len(all_acc_1) - 1]))
# print('Test Error: {0:1.3f}'.format(all_test_error_1[len(all_test_error_1) - 1]))
# print(
#     'TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(all_tp_1[len(all_tp_1) - 1],
#                                                                       all_fp_1[len(all_fp_1) - 1],
#                                                                       all_tn_1[len(all_tn_1) - 1],
#                                                                       all_fn_1[len(all_fn_1) - 1]))
# print('EOD: {0:1.4f}'.format(all_EOD_1[len(all_EOD_1) - 1]))
# print('SPD: {0:1.4f}'.format(all_SPD_1[len(all_SPD_1) - 1]))
# print('AOD: {0:1.4f}'.format(all_AOD_1[len(all_AOD_1) - 1]))
# print('F1: {0:1.3f}'.format(all_f1_1[len(all_f1_1) - 1]))
# print("")
# print('**********************')
# print('*       Female       *')
# print('**********************')
# print('Test Accuracy: {0:1.3f}'.format(all_f_acc_1[len(all_f_acc_1) - 1]))
# print('Test Error: {0:1.3f}'.format(all_F_ERR_1[len(all_F_ERR_1) - 1]))
# print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(all_F_TP_1[len(all_F_TP_1) - 1],
#                                                                         all_F_FP_1[len(all_F_FP_1) - 1],
#                                                                         all_F_TN_1[len(all_F_TN_1) - 1],
#                                                                         all_F_FN_1[len(all_F_FN_1) - 1]))
# print('F1: {0:1.3f}'.format(all_F_F1_1[len(all_F_F1_1) - 1]))
# print("")
# print('********************')
# print('*       Male       *')
# print('********************')
# print('Test Accuracy: {0:1.3f}'.format(all_m_acc_1[len(all_m_acc_1) - 1]))
# print('Test Error: {0:1.3f}'.format(all_M_ERR_1[len(all_M_ERR_1) - 1]))
# print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(all_M_TP_1[len(all_M_TP_1) - 1],
#                                                                         all_M_FP_1[len(all_M_FP_1) - 1],
#                                                                         all_M_TN_1[len(all_M_TN_1) - 1],
#                                                                         all_M_FN_1[len(all_M_FN_1) - 1]))
# print('F1: {0:1.3f}'.format(all_M_F1_1[len(all_M_F1_1) - 1]))
#
# print('Train Time: {0:1.2f}'.format(all_times_1[len(all_times_1) - 1]))
# print('AUROC: {0:1.2f}'.format(all_roc_1[len(all_roc_1) - 1]))
#
# print('\n')
# print('~' * 30)
# print('~         Client Two         ~')
# print('~' * 30)
# print('*******************')
# print('*       all       *')
# print('*******************')
# print('Test Accuracy: {0:1.3f}'.format(all_acc_2[len(all_acc_2) - 1]))
# print('Test Error: {0:1.3f}'.format(all_test_error_2[len(all_test_error_2) - 1]))
# print(
#     'TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(all_tp_2[len(all_tp_2) - 1],
#                                                                       all_fp_2[len(all_fp_2) - 1],
#                                                                       all_tn_2[len(all_tn_2) - 1],
#                                                                       all_fn_2[len(all_fn_2) - 1]))
# print('EOD: {0:1.4f}'.format(all_EOD_2[len(all_EOD_2) - 1]))
# print('SPD: {0:1.4f}'.format(all_SPD_2[len(all_SPD_2) - 1]))
# print('AOD: {0:1.4f}'.format(all_AOD_2[len(all_AOD_2) - 1]))
# print('F1: {0:1.3f}'.format(all_f1_2[len(all_f1_2) - 1]))
# print("")
# print('**********************')
# print('*       Female       *')
# print('**********************')
# print('Test Accuracy: {0:1.3f}'.format(all_f_acc_2[len(all_f_acc_2) - 1]))
# print('Test Error: {0:1.3f}'.format(all_F_ERR_2[len(all_F_ERR_2) - 1]))
# print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(all_F_TP_2[len(all_F_TP_2) - 1],
#                                                                         all_F_FP_2[len(all_F_FP_2) - 1],
#                                                                         all_F_TN_2[len(all_F_TN_2) - 1],
#                                                                         all_F_FN_2[len(all_F_FN_2) - 1]))
# print('F1: {0:1.3f}'.format(all_F_F1_2[len(all_F_F1_2) - 1]))
# print("")
# print('********************')
# print('*       Male       *')
# print('********************')
# print('Test Accuracy: {0:1.3f}'.format(all_m_acc_2[len(all_m_acc_2) - 1]))
# print('Test Error: {0:1.3f}'.format(all_M_ERR_2[len(all_M_ERR_2) - 1]))
# print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(all_M_TP_2[len(all_M_TP_2) - 1],
#                                                                         all_M_FP_2[len(all_M_FP_2) - 1],
#                                                                         all_M_TN_2[len(all_M_TN_2) - 1],
#                                                                         all_M_FN_2[len(all_M_FN_2) - 1]))
# print('F1: {0:1.3f}'.format(all_M_F1_2[len(all_M_F1_2) - 1]))
#
# print('Train Time: {0:1.2f}'.format(all_times_2[len(all_times_2) - 1]))
# print('AUROC: {0:1.2f}'.format(all_roc_2[len(all_roc_2) - 1]))
#
# print('\n')
# print('~' * 30)
# print('~        Global Model        ~')
# print('~' * 30)
# print('*******************')
# print('*       all       *')
# print('*******************')
# print('Test Accuracy: {0:1.3f}'.format(acc_all[len(acc_all) - 1]))
# print('Test Error: {0:1.3f}'.format(test_error_all[len(test_error_all) - 1]))
# print(
#     'TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(tp_all[len(tp_all) - 1], fp_all[len(fp_all) - 1],
#                                                                       tn_all[len(tn_all) - 1], fn_all[len(fn_all) - 1]))
# print('EOD: {0:1.4f}'.format(EOD_all[len(EOD_all) - 1]))
# print('SPD: {0:1.4f}'.format(SPD_all[len(SPD_all) - 1]))
# print('AOD: {0:1.4f}'.format(AOD_all[len(AOD_all) - 1]))
# print('F1: {0:1.3f}'.format(f1_all[len(f1_all) - 1]))
# print("")
# print('**********************')
# print('*       Female       *')
# print('**********************')
# print('Test Accuracy: {0:1.3f}'.format(F_ACC_all[len(F_ACC_all) - 1]))
# print('Test Error: {0:1.3f}'.format(F_ERR_all[len(F_ERR_all) - 1]))
# print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(F_TP_all[len(F_TP_all) - 1],
#                                                                         F_FP_all[len(F_FP_all) - 1],
#                                                                         F_TN_all[len(F_TN_all) - 1],
#                                                                         F_FN_all[len(F_FN_all) - 1]))
# print('F1: {0:1.3f}'.format(F_F1_all[len(F_F1_all) - 1]))
# print("")
# print('********************')
# print('*       Male       *')
# print('********************')
# print('Test Accuracy: {0:1.3f}'.format(M_ACC_all[len(M_ACC_all) - 1]))
# print('Test Error: {0:1.3f}'.format(M_ERR_all[len(M_ERR_all) - 1]))
# print('TP: {0:1.1f};\tFP: {1:1.1f};\tTN: {2:1.1f};\tFN {3:1.1f}'.format(M_TP_all[len(M_TP_all) - 1],
#                                                                         M_FP_all[len(M_FP_all) - 1],
#                                                                         M_TN_all[len(M_TN_all) - 1],
#                                                                         M_FN_all[len(M_FN_all) - 1]))
# print('F1: {0:1.3f}'.format(M_F1_all[len(M_F1_all) - 1]))
#
# print('Aggregate and Update Time: {0:1.2f}'.format(times_all[len(times_all) - 1]))
# print('AUROC: {0:1.2f}'.format(roc_all[len(roc_all) - 1]))