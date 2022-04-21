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
from fairtorch import DemographicParityLoss
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

warnings.filterwarnings("ignore")

m = "log-reg-two-fl"

# all_acc_1, all_f_acc_1, all_m_acc_1 = [], [], []
# all_test_error_1, all_F_ERR_1, all_M_ERR_1 = [], [], []
# all_tp_1, all_tn_1, all_fp_1, all_fn_1, all_f1_1 = [], [], [], [], []
# all_F_TP_1, all_F_FP_1, all_F_TN_1, all_F_FN_1, all_F_F1_1 = [], [], [], [], []
# all_M_TP_1, all_M_FP_1, all_M_TN_1, all_M_FN_1, all_M_F1_1 = [], [], [], [], []
# all_EOD_1, all_SPD_1, all_AOD_1 = [], [], []
# all_times_1, all_roc_1 = [], []
#
# all_acc_2, all_f_acc_2, all_m_acc_2 = [], [], []
# all_test_error_2, all_F_ERR_2, all_M_ERR_2 = [], [], []
# all_tp_2, all_tn_2, all_fp_2, all_fn_2, all_f1_2 = [], [], [], [], []
# all_F_TP_2, all_F_FP_2, all_F_TN_2, all_F_FN_2, all_F_F1_2 = [], [], [], [], []
# all_M_TP_2, all_M_FP_2, all_M_TN_2, all_M_FN_2, all_M_F1_2 = [], [], [], [], []
# all_EOD_2, all_SPD_2, all_AOD_2 = [], [], []
# all_times_2, all_roc_2 = [], []

clients = 2

#for i in range(1):
#    print('Round: ', i)


def train(config):
    all_spd_1 =[]
    all_spd_2 = []

    c1_loss_values, c1_test_loss_values = [], []
    c1_acc_values, c1_test_acc, c1_f_acc, c1_m_acc = [], [], [], []
    c1_EOD, c1_SPD, c1_AOD = [], [], []

    c2_loss_values, c2_test_loss_values = [], []
    c2_acc_values, c2_test_acc, c2_f_acc, c2_m_acc = [], [], [], []
    c2_EOD, c2_SPD, c2_AOD = [], [], []

    seed_everything(0)
    if m == "log-reg-two-fl":
        c1_model = LR_combo(input_size=9, vector_size=9, hidden_size=100)
        c2_model = LR_combo(input_size=9, vector_size=9, hidden_size=100)
        hnet = LR_HyperNet(vector_size=9, hidden_dim=100, num_hidden=config['h-num-hidden'])
        c1_l = config["c1-lr"]
        c2_l = config["c2-lr"]
        o_l = config["h-lr"]
    elif m == "neural-net-two-fl":
        c1_model = NN_combo(input_size=9, vector_size=9)
        c2_model = NN_combo(input_size=9, vector_size=9)
        hnet = NN_HyperNet(vector_size=9, hidden_dim=100, num_hidden=3)
        c1_l = .003
        c2_l = .005
        o_l = .002

    c1_model = c1_model.double()
    c2_model = c2_model.double()
    hnet = hnet.double()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            c1_model = nn.DataParallel(c1_model)
            c2_model = nn.DataParallel(c2_model)

    hnet = hnet.to(device)
    c1_model = c1_model.to(device)
    c2_model = c2_model.to(device)

    optimizer = torch.optim.Adam(hnet.parameters(), lr=o_l)
    c1_inner_optimizer = torch.optim.Adam(c1_model.parameters(), lr=c1_l)
    c2_inner_optimizer = torch.optim.Adam(c2_model.parameters(), lr=c2_l)

    loss = nn.BCEWithLogitsLoss(reduction='mean')  # binary logarithmic loss function

    data_train_1, data_test_1, data_train_2, data_test_2, features_1, features_2, decision_1, decision_2, d_test_1, d_test_2, encoders = get_data()

    torch.cuda.synchronize()
    for step in range(config["step"]):
        #Need to save model so client both have the same starting point
        torch.save(hnet.state_dict(), 'hnet_baseline_weights.pth')

        for c in range(clients):
            if c == 0:
                data_train = data_train_1
                data_test = data_test_1
                dp_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=config["alpha1"])
            else:
                data_train = data_train_2
                data_test = data_test_2
                dp_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=config["alpha2"])

            train_loader = DataLoader(data_train, shuffle=True, batch_size=256)
            test_loader = DataLoader(data_test, shuffle=False, batch_size=256)

            #load baseline
            hnet.load_state_dict(torch.load('hnet_baseline_weights.pth'))

            hnet.train()
            start_epoch = time.time()

            for epoch in range(config["ep"]):
                if c == 0:
                    model = c1_model
                    inner_optimizer = c1_inner_optimizer
                if c == 1:
                    model = c2_model
                    inner_optimizer = c2_inner_optimizer

                model.train()
                running_loss = 0.0
                correct = 0
                epoch_steps = 0
                for i, (x, y, s) in enumerate(train_loader):
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

                    y_, y_raw = model(x.to(device), context_only=False)
                    err = loss(y_raw.flatten(), y.to(device)) + dp_loss(x.float(), y_raw.float(), s.float())
                    err = err.mean()
                    running_loss += err.item() #* x.size(0)
                    epoch_steps += 1
                    # if i % 5 == 0:
                    #     print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / epoch_steps))
                    #     running_loss = 0.0

                    err.backward()
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
                correct = 0.0
                test_steps = 0
                with torch.no_grad():
                    for i, (x, y, s) in enumerate(test_loader):
                        pred, pred_raw = model(x.to(device), context_only=False)
                        test_err = loss(pred_raw.flatten(), y.to(device))
                        test_err = test_err.mean()
                        running_loss_test += test_err.item() * x.size(0)
                        test_steps += 1
                        preds = pred.detach().cpu().round().reshape(1, len(pred))
                        predictions.extend(preds.flatten().numpy())
                        correct += (preds.eq(y)).sum().item()

                        predicted_prediction = preds.type(torch.IntTensor).numpy().reshape(-1)
                        labels_pred = y.type(torch.IntTensor).numpy().reshape(-1)

                        TP, FP, FN, TN, f_tp, f_fp, f_tn, f_fn, m_tp, m_fp, m_tn, m_fn = confusion_matrix(s,
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

                f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, aod, eod, spd = metrics(
                    TP, FP, FN, f_tp, f_fp, f_fn, m_tp, m_fp, m_fn, TN, f_tn, m_tn)
                if c == 0:
                    c1_SPD.append(spd)
                elif c == 1:
                    c2_SPD.append(spd)

            optimizer.zero_grad()
            final_state = model.state_dict()

            # calculating delta theta
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

            # calculating phi gradient
            hnet_grads = torch.autograd.grad(
                list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
            )

            # update hnet weights
            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g

            # torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
            optimizer.step()

        all_spd_1.append(abs(c1_SPD[len(c1_SPD) - 1]))
        all_spd_2.append(abs(c2_SPD[len(c2_SPD) - 1]))

    tune.report(accuracy=accuracy, SPD=(all_spd_1[len(all_spd_1) - 1] + all_spd_2[len(all_spd_2) - 1]))


def main(num_samples = 10, max_num_epochs = 50, gpus_per_trial = 1):
    config = {
        "h-num-hidden": tune.sample_from(lambda _: np.random.randint(1, 11)),
        "c1-lr": tune.loguniform(1e-5, 1e-1),
        "c2-lr": tune.loguniform(1e-5, 1e-1),
        "h-lr": tune.loguniform(1e-5, 1e-1),
        "step": tune.sample_from(lambda _: np.random.randint(5, 21)),
        "ep": tune.choice([10, 20, 25, 50, 100]),
        "alpha1": tune.sample_from(lambda _: np.random.randint(1, 101)),
        "alpha2": tune.sample_from(lambda _: np.random.randint(1, 101))
    }

    scheduler = ASHAScheduler(
        metric="SPD",
        mode="min",
        max_t = max_num_epochs,
        grace_period= 10,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["accuracy", "SPD"],
        metric="SPD",
        mode="min"
    )

    result = tune.run(
        partial(train),
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("SPD", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))
    print("Best trial SPD: {0:1.3f}".format( best_trial.last_result["SPD"]))
    best_trial_acc = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial_acc.config))
    print("Best trial final validation accuracy: {}".format(best_trial_acc.last_result["accuracy"]))
    print("Best trial SPD: {0:1.3f}".format(best_trial_acc.last_result["SPD"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=100, max_num_epochs=100, gpus_per_trial=.1)
