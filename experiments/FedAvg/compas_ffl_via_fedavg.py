import time
import warnings
import torch
import numpy as np
import random
import torch.nn as nn
from experiments.adult.utils import seed_everything, metrics, TP_FP_TN_FN
from models import LR, NN
from experiments.adult.node import BaseNodes

warnings.filterwarnings("ignore")

no_cuda=False
gpus='5'
device = torch.cuda.set_device(5)

def train_no_dp(model, train_loader, loss, optimizer):
    model.train()
    correct = 0
    running_loss = 0.0

    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_, y_raw = model(x.to(device))
        err = loss(y_raw.flatten(), y.to(device))
        err = err.mean()
        running_loss += err.item() * x.size(0)
        err.backward()
        optimizer.step()

        preds = y_.round().reshape(1, len(y_))
        correct += (preds.eq(y)).sum().item()

    return (running_loss / len(train_loader)), (100 * correct/ len(train_loader.dataset))

def validation_no_dp(model, num_samples, loss):
    model.eval()
    predictions = []
    running_loss = 0.0
    running_correct = 0
    running_samples = 0
    pred_client = []
    true_client = []
    queries_client = []
    len_test_loaders = 0
    aod = []
    eod = []
    spd = []
    f1 = []
    f1_f = []
    f1_m = []
    a = []
    f_a = []
    m_a = []

    for node_id in range(num_samples):
        curr_data = nodes.test_loaders[node_id]
        len_test_loaders += len(nodes.test_loaders[node_id])
        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.FloatTensor)) for t in batch)
            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            pred = model(x)
            pred_prob = torch.sigmoid(pred)
            pred_thresh = (pred_prob > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())
            running_loss += loss(pred, y.unsqueeze(1)).item()

            correct = torch.eq(pred_thresh, y.unsqueeze(1)).type(torch.LongTensor)
            running_correct += torch.count_nonzero(correct).item()

            running_samples += len(y)

        tp, fp, tn, fn = TP_FP_TN_FN(queries_client, pred_client, true_client, fair="none")
        f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, AOD, EOD, SPD = metrics(tp, fp, tn, fn)
        f1.append(f1_score_prediction)
        f1_f.append(f1_female)
        f1_m.append(f1_male)
        a.append(accuracy)
        f_a.append(f_acc)
        m_a.append(m_acc)
        aod.append(AOD)
        eod.append(EOD)
        spd.append(SPD)

    running_loss /= len_test_loaders
    return predictions, running_loss, np.mean(f1), np.mean(f1_f), np.mean(f1_m), np.mean(a), np.mean(f_a), np.mean(m_a), np.mean(aod), np.mean(eod), np.mean(spd)

def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples, number_of_clients_per_round):
    sampled = []

    with torch.no_grad():
        for i in range(number_of_clients_per_round):
            node_id = random.choice(range(number_of_samples))
            sampled.append(node_id)

            model_dict[name_of_models[node_id]].fc1.weight.data = main_model.fc1.weight.data.clone()
            model_dict[name_of_models[node_id]].fc1.bias.data = main_model.fc1.bias.data.clone()

            # model_dict[name_of_models[node_i]].fc2.weight.data = main_model.fc2.weight.data.clone()
            # model_dict[name_of_models[node_i]].fc2.bias.data = main_model.fc2.bias.data.clone()
            #
            # model_dict[name_of_models[node_i]].fc3.weight.data = main_model.fc3.weight.data.clone()
            # model_dict[name_of_models[node_i]].fc3.bias.data = main_model.fc3.bias.data.clone()
            #
            # model_dict[name_of_models[node_i]].fc4.weight.data = main_model.fc4.weight.data.clone()
            # model_dict[name_of_models[node_i]].fc4.bias.data = main_model.fc4.bias.data.clone()
            #
            # model_dict[name_of_models[node_i]].fc5.weight.data = main_model.fc5.weight.data.clone()
            # model_dict[name_of_models[node_i]].fc5.bias.data = main_model.fc5.bias.data.clone()

        return model_dict, sampled

def create_model_optimizer_criterion_dict(number_of_samples):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)
        if m == "log-reg":
            model_info = LR(input_size=10)
        elif m == "neural-net":
            model_info = NN(input_size=10)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.Adam(model_info.parameters(), lr=learning_rate, weight_decay=wd)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.BCEWithLogitsLoss(reduction='mean')
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict

def start_train_end_node_process_print_some(sampled):
    for i, client in enumerate(sampled):
        model = model_dict[name_of_models[client]]
        criterion = criterion_dict[name_of_criterions[client]]
        optimizer = optimizer_dict[name_of_optimizers[client]]

        for epoch in range(num_epoch):
            batch = next(iter(nodes.train_loaders[client]))
            x, y = tuple((t.type(torch.FloatTensor)) for t in batch)

            model.train()
            optimizer.zero_grad()

            # train and update local
            pred = model(x)

            err = criterion(pred, y.unsqueeze(1))
            err.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()


def get_averaged_weights(model_dict, sampled):
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
        for i, client in enumerate(sampled):
            fc1_mean_weight += model_dict[name_of_models[client]].fc1.weight.data.clone()
            fc1_mean_bias += model_dict[name_of_models[client]].fc1.bias.data.clone()

            # fc2_mean_weight += model_dict[name_of_models[client]].fc2.weight.data.clone()
            # fc2_mean_bias += model_dict[name_of_models[client]].fc2.bias.data.clone()
            #
            # fc3_mean_weight += model_dict[name_of_models[client]].fc3.weight.data.clone()
            # fc3_mean_bias += model_dict[name_of_models[client]].fc3.bias.data.clone()
            #
            # fc4_mean_weight += model_dict[name_of_models[client]].fc4.weight.data.clone()
            # fc4_mean_bias += model_dict[name_of_models[client]].fc4.bias.data.clone()
            #
            # fc5_mean_weight += model_dict[name_of_models[client]].fc5.weight.data.clone()
            # fc5_mean_bias += model_dict[name_of_models[client]].fc5.bias.data.clone()

        fc1_mean_weight = fc1_mean_weight / number_of_samples
        fc1_mean_bias = fc1_mean_bias / number_of_samples
        #
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


def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model,model_dict, sampled):
    #fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias, fc4_mean_weight, fc4_mean_bias, fc5_mean_weight, fc5_mean_bias = get_averaged_weights(model_dict, number_of_samples=number_of_samples)
    fc1_mean_weight, fc1_mean_bias = get_averaged_weights(model_dict, sampled)

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
number_of_samples = 4
nodes = BaseNodes("compas", number_of_samples, 256, 2, "none")

results_all = []
acc_all, F_ACC_all, M_ACC_all = [], [], []
f1_all = []
F_F1_all = []
M_F1_all = []
EOD_all, SPD_all, AOD_all = [], [], []
times_all, roc_all = [], []

m = "log-reg"

if m == "log-reg":
    main_model = LR(input_size = 10)
    learning_rate = .003
    num_epoch = 50         # local
    wd = 0.00001
else:
    main_model = NN(input_size = 10)
    learning_rate = .005
    num_epoch = 50         # local
    wd = 0.00001

loss = nn.BCEWithLogitsLoss(reduction='mean')
main_criterion = nn.BCEWithLogitsLoss()

model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(number_of_samples)

name_of_models = list(model_dict.keys())
name_of_optimizers = list(optimizer_dict.keys())
name_of_criterions = list(criterion_dict.keys())

print('~'*22)
print('~~~ Start Training ~~~')
print('~'*22,'\n')
results_main_model = []

num_client_sample_per_round = 2

for i in range(1000):
    start = time.time()
    model_dict, sampled = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples, num_client_sample_per_round)
    start_train_end_node_process_print_some(sampled)
    main_model = set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, sampled)
    end = time.time()
    predictions, running_loss_test, f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, aod, eod, spd = validation_no_dp(main_model,number_of_samples, main_criterion)

    times_all.append(end - start)
    f1_all.append(f1_score_prediction)
    F_F1_all.append(f1_female)
    M_F1_all.append(f1_male)
    AOD_all.append(aod)
    SPD_all.append(spd)
    EOD_all.append(eod)

    acc_all.append(accuracy)
    F_ACC_all.append(f_acc)
    M_ACC_all.append(m_acc)
    
    print("\nIteration", str(i + 1), ": main_model accuracy on all test data: {:7.4f}".format(accuracy))
    print("\n")

print('\n')

print('~'*30)
print('~        Global Model        ~')
print('~'*30)
print('*******************')
print('*       all       *')
print('*******************')
print('Test Accuracy: {0:1.3f}'.format(acc_all[len(acc_all) - 1]))
print('EOD: {0:1.4f}'.format(EOD_all[len(EOD_all) - 1]))
print('SPD: {0:1.4f}'.format(SPD_all[len(SPD_all) - 1]))
print('AOD: {0:1.4f}'.format(AOD_all[len(AOD_all) - 1]))
print('F1: {0:1.3f}'.format(f1_all[len(f1_all) - 1]))
print("")
print('**********************')
print('*       Female       *')
print('**********************')
print('Test Accuracy: {0:1.3f}'.format(F_ACC_all[len(F_ACC_all) - 1]))
print('F1: {0:1.3f}'.format(F_F1_all[len(F_F1_all) - 1]))
print("")
print('********************')
print('*       Male       *')
print('********************')
print('Test Accuracy: {0:1.3f}'.format(M_ACC_all[len(M_ACC_all) - 1]))
print('F1: {0:1.3f}'.format(M_F1_all[len(M_F1_all) - 1]))

print('Aggregate and Update Time: {0:1.2f}'.format(times_all[len(times_all) - 1]))
