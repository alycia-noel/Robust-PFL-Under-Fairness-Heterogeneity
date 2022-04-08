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

clients = 2


c1_loss_values, c1_test_loss_values = [], []
c1_acc_values, c1_test_acc, c1_f_acc, c1_m_acc = [], [], [], []
c1_test_error, c1_f_err, c1_m_err = [], [], []
c1_tp, c1_tn, c1_fp, c1_fn = [], [], [], []
c1_f_tp, c1_f_tn, c1_f_fp, c1_f_fn = [], [], [], []
c1_m_tp, c1_m_tn, c1_m_fp, c1_m_fn = [], [], [], []
c1_times = []
c1_f1, c1_f_f1, c1_m_f1 = [], [], []
c1_EOD, c1_SPD, c1_AOD = [], [], []
c1_results, c1_final_results, c1_final_final_results, c1_all_results = [], [], [], []

c2_loss_values, c2_test_loss_values = [], []
c2_acc_values, c2_test_acc, c2_f_acc, c2_m_acc = [], [], [], []
c2_test_error, c2_f_err, c2_m_err = [], [], []
c2_tp, c2_tn, c2_fp, c2_fn = [], [], [], []
c2_f_tp, c2_f_tn, c2_f_fp, c2_f_fn = [], [], [], []
c2_m_tp, c2_m_tn, c2_m_fp, c2_m_fn = [], [], [], []
c2_times = []
c2_f1, c2_f_f1, c2_m_f1 = [], [], []
c2_EOD, c2_SPD, c2_AOD = [], [], []
c2_results, c2_final_results, c2_final_final_results, c2_all_results = [], [], [], []

def train(config):
    all_acc_1 = []
    all_acc_2 = []

    final_acc_1 = 0
    final_acc_2 = 0
    seed_everything(0)
    if m == "log-reg-two-fl":
        c1_model = LR_combo(input_size=10, vector_size=10, hidden_size=100)
        c2_model = LR_combo(input_size=10, vector_size=10, hidden_size=100)
        hnet = LR_HyperNet(vector_size=10, hidden_dim=100, num_hidden=config['h-num-hidden'])
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

    loss = nn.BCELoss(reduction='mean')  # binary logarithmic loss function

    data_train_1, data_test_1, data_train_2, data_test_2, features_1, features_2, decision_1, decision_2, d_test_1, d_test_2, encoders = get_data()

    torch.cuda.synchronize()
    for step in range(config["step"]):
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

            # load baseline
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
                    err = loss(y_.flatten(), y.to(device))
                    err = err.mean()
                    running_loss += err.item()  # * x.size(0)
                    epoch_steps += 1

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
                    for i, (x, y) in enumerate(test_loader):
                        pred = model(x.to(device), context_only=False)
                        test_err = loss(pred.flatten(), y.to(device))
                        test_err = test_err.mean()
                        running_loss_test += test_err.item() * x.size(0)
                        test_steps += 1
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

                f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, error, f_err, m_err, aod, eod, spd = metrics(
                    TP, FP, FN, f_tp, f_fp, f_fn, m_tp, m_fp, m_fn, TN, f_tn, m_tn)

                if c == 0:
                    c1_test_acc.append(accuracy)
                elif c == 1:
                    c2_test_acc.append(accuracy)

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

        all_acc_1.append(c1_test_acc[len(c1_test_acc) - 1])
        all_acc_2.append(c2_test_acc[len(c2_test_acc) - 1])

    tune.report(accuracy=(all_acc_1[len(all_acc_1) - 1] + all_acc_2[len(all_acc_2) - 1]))


def main(num_samples, max_num_epochs, gpus_per_trial):
    config = {
        "h-num-hidden" : tune.sample_from(lambda _: np.random.randint(2, 10)),
        "c1-lr": tune.loguniform(1e-5, 1e-1),
        "c2-lr": tune.loguniform(1e-5, 1e-1),
        "h-lr": tune.loguniform(1e-5, 1e-1),
        "step": tune.sample_from(lambda _: np.random.randint(5, 21)),
        "ep": tune.sample_from(lambda _: np.random.randint(10, 101)),
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["accuracy"],
        metric="accuracy",
        mode="max"
    )

    result = tune.run(
        partial(train),
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=100, max_num_epochs=50, gpus_per_trial=.1)



