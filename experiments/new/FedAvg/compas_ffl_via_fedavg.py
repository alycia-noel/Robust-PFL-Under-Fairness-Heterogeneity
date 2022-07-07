import os
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3, 4, 5, 6, 7"
import argparse
import logging
import random
from collections import defaultdict
import torch
import pandas as pd
import torch.utils.data
from tqdm import trange
from models import LR, Constraint
from experiments.new.cFHN.node import BaseNodes
from experiments.new.cFHN.utils import seed_everything, set_logger, TP_FP_TN_FN, metrics
import numpy as np

def eval_model(nodes, num_nodes, model_dict, name_of_models, device, which_position):
    curr_results, preds, true, f1, a, aod, eod, spd = evaluate(nodes, num_nodes, model_dict, name_of_models, device, which_position)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]
    return curr_results, avg_acc, all_acc, f1, aod, eod, spd

@torch.no_grad()
def evaluate(nodes, num_nodes, model_dict, name_of_models, device, which_position):
    results = defaultdict(lambda: defaultdict(list))
    preds = []
    true = []
    f1, a, aod, eod, spd = [], [], [], [], []

    for node_id in range(num_nodes):

        pred_client = []
        true_client = []
        queries_client = []

        model = model_dict[name_of_models[node_id]]

        model.to(device)

        running_loss, running_correct, running_samples = 0, 0, 0

        curr_data = nodes.test_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
            s = x[:, which_position].to(device)

            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            pred, m_mu_q = model(x, s, y)
            pred_thresh = (pred > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())

            correct = torch.eq(pred_thresh, y.unsqueeze(1)).type(torch.cuda.LongTensor)
            running_correct += torch.count_nonzero(correct).item()

            running_samples += len(y)

        tp, fp, tn, fn = TP_FP_TN_FN(queries_client, pred_client, true_client, which_position)
        print(tp, fp, tn, fn)
        f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, AOD, EOD, SPD = metrics(tp, fp, tn, fn)
        f1.append(f1_score_prediction)
        a.append(accuracy)
        aod.append(AOD)
        eod.append(EOD)
        spd.append(SPD)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples
        preds.append(pred_client)
        true.append(true_client)

    return results, preds, true, f1, a, aod, eod, spd

def send_main_model_to_nodes_and_update_model_dict(final, main_model, model_dict, number_of_samples, number_of_clients_per_round, name_of_models):
    sampled = []

    with torch.no_grad():
        for i in range(number_of_clients_per_round):
            if final == 'no':
                node_id = random.choice(range(number_of_samples))
            elif final == 'yes':
                node_id = i
            sampled.append(node_id)

            model_dict[name_of_models[node_id]].fc1.weight.data = main_model.fc1.weight.data.clone()
            model_dict[name_of_models[node_id]].fc1.bias.data = main_model.fc1.bias.data.clone()

        return model_dict, sampled


def create_model_optimizer_criterion_dict(number_of_samples, fair, alpha, m, learning_rate, wd, num_features):
    model_dict = dict()
    optimizer_dict = dict()
    constraint_dict = dict()
    criterion_dict = dict()
    fair_loss_dict = dict()
    combo_param_dict = dict()

    for i in range(number_of_samples):
        fair_loss_name = "fair_loss" + str(i)
        if fair == "dp":
            fair_loss_info = 'dp'
        elif fair == "eo":
            fair_loss_info = 'eo'
        elif fair == "both":
            if i % 2 == 0:
                fair_loss_info = 'dp'
            else:
                fair_loss_info = 'eo'
        else:
            fair_loss_info = 'none'
        fair_loss_dict.update({fair_loss_name: fair_loss_info})


        model_name = "model" + str(i)
        model_info = LR(input_size=num_features, bound=.05, fairness=fair_loss_info)
        model_dict.update({model_name: model_info})

        constraint_name = "constraint" + str(i)
        constraint_info = Constraint(fair=fair_loss_info)
        constraint_dict.update({constraint_name: constraint_info})

        combo_param_name = "combo-param" + str(i)
        if fair == 'none':
            combo_param_info = list(model_info.parameters())
        else:
            combo_param_info = list(model_info.parameters()) + list(constraint_info.parameters())
        combo_param_dict.update({combo_param_name: combo_param_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.Adam(combo_param_info, lr=learning_rate, weight_decay=wd)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.BCELoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict, fair_loss_dict, combo_param_dict, constraint_dict


def start_train_end_node_process_print_some(alphas, combo_params_dict, name_of_combo_params, device, sampled, constraint_dict, name_of_constraint, model_dict, name_of_models, criterion_dict, name_of_criterions, optimizer_dict, name_of_optimizers, fair_loss_dict, name_of_fair_loss, num_epoch, nodes, fair, which_position):
    for i, client in enumerate(sampled):
        model = model_dict[name_of_models[client]]
        criterion = criterion_dict[name_of_criterions[client]]
        constraint = constraint_dict[name_of_constraint[client]]
        optimizer = optimizer_dict[name_of_optimizers[client]]
        combo_params = combo_params_dict[name_of_combo_params[client]]

        if fair != "none":
            fair_loss = fair_loss_dict[name_of_fair_loss[client]]
        else:
            fair_loss = 'none'

        if fair_loss == 'dp':
            alpha = alphas[0]
        elif fair_loss == 'eo':
            alpha = alphas[1]

        queries_client = []
        pred_client = []
        true_client =[]
        for epoch in range(num_epoch):
            batch = next(iter(nodes.train_loaders[client]))
            x, y = tuple((t.type(torch.FloatTensor)) for t in batch)
            s = x[:, which_position]

            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            model.train()
            optimizer.zero_grad()

            # train and update local
            pred, m_mu_q = model(x, s, y)
            pred_thresh = (pred > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())

            if fair_loss == 'none':
                err = criterion(pred, y.unsqueeze(1))
            else:
                err = criterion(pred, y.unsqueeze(1)) + (alpha * constraint(m_mu_q))

            err.backward()
            if fair != 'none':
                combo_params[len(combo_params) - 1].grad.data = -1 * combo_params[len(combo_params) - 1].grad.data

            optimizer.step()

        # tp, fp, tn, fn = TP_FP_TN_FN(queries_client, pred_client, true_client, which_position)
        #
        # f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, AOD, EOD, SPD = metrics(tp, fp, tn, fn)
        # print(AOD, EOD, SPD)


def get_averaged_weights(ratios, model_dict, sampled, name_of_models, number_of_samples):
    fc1_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc1.weight.shape)
    fc1_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc1.bias.shape)

    with torch.no_grad():
        for i, client in enumerate(sampled):
            fc1_mean_weight += ratios[client]*(model_dict[name_of_models[client]].fc1.weight.data.clone())
            fc1_mean_bias += ratios[client]*(model_dict[name_of_models[client]].fc1.bias.data.clone())

        fc1_mean_weight = fc1_mean_weight / number_of_samples
        fc1_mean_bias = fc1_mean_bias / number_of_samples

    return fc1_mean_weight, fc1_mean_bias


def set_averaged_weights_as_main_model_weights_and_update_main_model(ratios, main_model, model_dict, sampled, name_of_models, number_of_samples):
    fc1_mean_weight, fc1_mean_bias = get_averaged_weights(ratios, model_dict, sampled, name_of_models, number_of_samples)

    with torch.no_grad():
        main_model.fc1.weight.data = fc1_mean_weight.data.clone()
        main_model.fc1.bias.data = fc1_mean_bias.data.clone()

    return main_model

def train(device, data_name,model_name,classes_per_node,num_nodes,steps,lr,wd,bs, alpha,fair, which_position, inner_steps):
    seed_everything(0)

    nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node)

    num_features = len(nodes.features)

    total_data_length = 0
    client_data_length = []
    ratios = []
    for i in range(num_nodes):
        total_data_length += len(nodes.train_loaders[i])
        client_data_length.append(len(nodes.train_loaders[i]))

    for i in range(num_nodes):
        ratios.append(client_data_length[i]/total_data_length)

    main_model = LR(input_size=num_features, bound=.05, fairness='none')

    model_dict, optimizer_dict, criterion_dict, fair_loss_dict, combo_param_dict, constraint_dict = create_model_optimizer_criterion_dict(num_nodes, fair, alpha, model_name, lr, wd, num_features)

    name_of_models = list(model_dict.keys())
    name_of_optimizers = list(optimizer_dict.keys())
    name_of_criterions = list(criterion_dict.keys())
    name_of_fair_loss = list(fair_loss_dict.keys())
    name_of_combo_param = list(combo_param_dict.keys())
    name_of_constraint = list(constraint_dict.keys())


    print('~' * 22)
    print('~~~ Start Training ~~~')
    print('~' * 22, '\n')

    step_iter = trange(steps)

    num_client_sample_per_round = 1

    for step in step_iter:
        model_dict, sampled = send_main_model_to_nodes_and_update_model_dict('no',main_model, model_dict, num_nodes, num_client_sample_per_round, name_of_models)
        start_train_end_node_process_print_some(alpha, combo_param_dict, name_of_combo_param, device, sampled, constraint_dict, name_of_constraint, model_dict, name_of_models, criterion_dict, name_of_criterions, optimizer_dict, name_of_optimizers, fair_loss_dict, name_of_fair_loss, inner_steps, nodes, fair, which_position)
        main_model = set_averaged_weights_as_main_model_weights_and_update_main_model(ratios, main_model, model_dict, sampled, name_of_models, num_client_sample_per_round)

    model_dict, sampled = send_main_model_to_nodes_and_update_model_dict('yes',main_model, model_dict, num_nodes, num_nodes, name_of_models)
    curr_results, avg_acc, all_acc, f1, aod, eod, spd = eval_model(nodes, num_nodes, model_dict, name_of_models, device, which_position)
    logging.info(f"\n\nFinal Results |  AVG Acc: {avg_acc:.4f}")
    for i in range(num_nodes):
        print("\nClient", i + 1)
        print(i)
        print(f"Acc: {all_acc[i]:.4f}, F1: {f1[i]:.4f}, AOD: {aod[i]:.4f}, EOD: {eod[i]:.4f}, SPD: {spd[i]:.4f}")


def main():
    pd.set_option('display.float_format', lambda x: '%.1f' % x)

    parser = argparse.ArgumentParser(description="Fair Hypernetworks")

    parser.add_argument("--data_name", type=str, default="adult", choices=["adult", "compas"], help="choice of dataset")
    parser.add_argument("--model_name", type=str, default="LR", choices=["NN", "LR"], help="choice of model")
    parser.add_argument("--num_nodes", type=int, default=4, help="number of simulated clients")
    parser.add_argument("--num_steps", type=int, default=5000)
    parser.add_argument("--inner_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-10, help="weight decay")
    parser.add_argument("--gpu", type=int, default=4, help="gpu device ID")
    parser.add_argument("--fair", type=str, default="both", choices=["none", "eo", "dp", "both"],
                        help="whether to use fairness of not.")
    parser.add_argument("--alpha", type=int, default=[100,100], help="fairness/accuracy trade-off parameter")
    parser.add_argument("--which_position", type=int, default=8, choices=[5, 8],
                        help="which position the sensitive attribute is in. 5: compas, 8: adult")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count()
    set_logger()

    device = 'cuda:4'
    args.class_per_node = 2

    train(
        device=device,
        data_name=args.data_name,
        model_name = args.model_name,
        classes_per_node = args.class_per_node,
        num_nodes=args.num_nodes,
        steps=args.num_steps,
        lr= args.lr,
        wd=args.wd,
        bs=args.batch_size,
        alpha=args.alpha,
        fair=args.fair,
        which_position=args.which_position,
        inner_steps=args.inner_steps
    )

if __name__ == "__main__":
    main()

