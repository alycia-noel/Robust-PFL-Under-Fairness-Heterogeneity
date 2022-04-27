import time
import argparse
import warnings
import torch
import numpy as np
import random
import logging
from tqdm import trange
import torch.nn as nn
from collections import defaultdict
from experiments.new.utils import seed_everything, metrics, TP_FP_TN_FN, set_logger, get_device
from experiments.new.models import LR
from experiments.new.node import BaseNodes
from fairtorch import DemographicParityLoss, EqualiedOddsLoss
warnings.filterwarnings("ignore")
import pandas as pd

def eval_model(nodes, model, num_nodes, loss, device, fair, fair_loss, which_position):
    curr_results, queries, pred, true, f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd = evaluate(nodes, model, num_nodes, loss, device, fair, fair_loss, which_position)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]
    all_loss = [val['loss'] for val in curr_results.values()]

    tp, fp, tn, fn = TP_FP_TN_FN(queries, pred, true, which_position)

    f1_score_prediction, f1_female, f1_male, accuracy, f_acc, m_acc, AOD, EOD, SPD = metrics(tp, fp, tn, fn)

    return curr_results, avg_loss, avg_acc, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd, AOD, EOD, SPD

@torch.no_grad()
def evaluate(nodes, model, num_samples, loss, device, fair, fair_loss, which_position):
    model.eval()
    results = defaultdict(lambda: defaultdict(list))
    preds = []
    true = []
    queries = []
    f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd = [], [], [], [], [], [], [], [], []

    for node_id in range(num_samples):
        pred_client = []
        true_client = []
        queries_client = []

        running_loss, running_correct, running_samples = 0, 0, 0

        curr_data = nodes.test_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.FloatTensor)) for t in batch)
            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            pred = model(x)
            pred_prob = torch.sigmoid(pred)
            pred_thresh = (pred_prob > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())

            running_loss += loss(pred, y.unsqueeze(1)).item()

            correct = torch.eq(pred_thresh, y.unsqueeze(1)).type(torch.cuda.LongTensor)
            running_correct += torch.count_nonzero(correct).item()

            running_samples += len(y)

        tp, fp, tn, fn = TP_FP_TN_FN(queries_client, pred_client, true_client, which_position)

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
        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples
        queries.extend(queries_client)
        preds.extend(pred_client)
        true.extend(true_client)

    return results, queries, preds, true, f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd


def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples, number_of_clients_per_round, name_of_models):
    sampled = []

    with torch.no_grad():
        for i in range(number_of_clients_per_round):
            node_id = random.choice(range(number_of_samples))
            sampled.append(node_id)

            model_dict[name_of_models[node_id]].fc1.weight.data = main_model.fc1.weight.data.clone()
            model_dict[name_of_models[node_id]].fc1.bias.data = main_model.fc1.bias.data.clone()

        return model_dict, sampled


def create_model_optimizer_criterion_dict(number_of_samples, fair, alpha, m, learning_rate, wd, num_features):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()
    fair_loss_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)
        if m == "LR":
            model_info = LR(input_size=num_features)

        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.Adam(model_info.parameters(), lr=learning_rate, weight_decay=wd)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.BCEWithLogitsLoss(reduction='mean')
        criterion_dict.update({criterion_name: criterion_info})

        if fair == "dp":
            fair_loss_name = "fair_loss" + str(i)
            fair_loss_info = DemographicParityLoss(sensitive_classes=[0, 1], alpha=alpha)
            fair_loss_dict.update({fair_loss_name: fair_loss_info})
        elif fair == "eo":
            fair_loss_name = "fair_loss" + str(i)
            fair_loss_info = EqualiedOddsLoss(sensitive_classes=[0, 1], alpha=alpha)
            fair_loss_dict.update({fair_loss_name: fair_loss_info})

    return model_dict, optimizer_dict, criterion_dict, fair_loss_dict


def start_train_end_node_process_print_some(device,sampled, model_dict, name_of_models, criterion_dict, name_of_criterions, optimizer_dict, name_of_optimizers, fair_loss_dict, name_of_fair_loss, num_epoch, nodes, fair, which_position):
    for i, client in enumerate(sampled):
        model = model_dict[name_of_models[client]]
        criterion = criterion_dict[name_of_criterions[client]]
        optimizer = optimizer_dict[name_of_optimizers[client]]
        if fair != "none":
            fair_loss = fair_loss_dict[name_of_fair_loss[client]]

        for epoch in range(num_epoch):
            batch = next(iter(nodes.train_loaders[client]))
            x, y = tuple((t.type(torch.FloatTensor)) for t in batch)
            s = x[:, which_position]

            model.train()
            optimizer.zero_grad()

            # train and update local
            pred = model(x)

            if fair == 'none':
                err = criterion(pred, y.unsqueeze(1))
            else:
                fair = fair_loss(x, pred, s, y)
                err = criterion(pred, y.unsqueeze(1)) + fair

            err.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()


def get_averaged_weights(model_dict, sampled, name_of_models, number_of_samples):
    fc1_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].fc1.weight.shape)
    fc1_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].fc1.bias.shape)

    with torch.no_grad():
        for i, client in enumerate(sampled):
            fc1_mean_weight += model_dict[name_of_models[client]].fc1.weight.data.clone()
            fc1_mean_bias += model_dict[name_of_models[client]].fc1.bias.data.clone()

        fc1_mean_weight = fc1_mean_weight / number_of_samples
        fc1_mean_bias = fc1_mean_bias / number_of_samples

    return fc1_mean_weight, fc1_mean_bias


def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, sampled, name_of_models, number_of_samples):
    fc1_mean_weight, fc1_mean_bias = get_averaged_weights(model_dict, sampled, name_of_models, number_of_samples)

    with torch.no_grad():
        main_model.fc1.weight.data = fc1_mean_weight.data.clone()
        main_model.fc1.bias.data = fc1_mean_bias.data.clone()

    return main_model

def train(device, data_name,model_name,classes_per_node,num_nodes,steps,lr,wd,bs, alpha,fair, which_position, inner_steps):
    seed_everything(0)

    nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node)

    num_features = len(nodes.features)

    times_all, roc_all = [], []

    if model_name == "LR":
        main_model = LR(input_size=num_features)

    loss = nn.BCEWithLogitsLoss(reduction='mean')

    model_dict, optimizer_dict, criterion_dict, fair_loss_dict = create_model_optimizer_criterion_dict(num_nodes, fair, alpha, model_name, lr, wd, num_features)

    name_of_models = list(model_dict.keys())
    name_of_optimizers = list(optimizer_dict.keys())
    name_of_criterions = list(criterion_dict.keys())
    name_of_fair_loss = list(fair_loss_dict.keys())

    print('~' * 22)
    print('~~~ Start Training ~~~')
    print('~' * 22, '\n')
    step_iter = trange(steps)

    num_client_sample_per_round = 1

    for step in step_iter:
        start = time.time()
        model_dict, sampled = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, num_nodes, num_client_sample_per_round, name_of_models)
        start_train_end_node_process_print_some(device,sampled, model_dict, name_of_models, criterion_dict, name_of_criterions, optimizer_dict, name_of_optimizers, fair_loss_dict, name_of_fair_loss, inner_steps, nodes, fair, which_position)
        main_model = set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, sampled, name_of_models, num_client_sample_per_round)
        end = time.time()

        times_all.append(end - start)
        if step % 49 == 0 or step == 1999 or step == 0:
            curr_results, avg_loss, avg_acc, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd, _, _, _ = eval_model(
                nodes=nodes, model=main_model, num_nodes=num_nodes, loss=loss, device=device, fair=fair,
                fair_loss=fair_loss_dict, which_position=which_position)
            logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

    curr_results, avg_loss, avg_acc, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd, aod_all, eod_all, spd_all = eval_model(
        nodes=nodes, model=main_model, num_nodes=num_nodes, loss=loss, device=device, fair=fair,
        fair_loss=fair_loss_dict, which_position=which_position)
    logging.info(f"\n\nFinal Results | AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}, AVG AOD: {aod_all:.4f}, AVG EOD: {eod_all:.4f}, AVG SPD: {spd_all}")
    for i in range(num_nodes):
        print("\nClient", i + 1)
        print(i)
        print(f"Acc: {all_acc[i]:.4f}, F Acc: {f_a[i]:.4f}, M Acc: {m_a[i]:.4f}, F1: {f1[i]:.4f}, AOD: {aod[i]:.4f}, EOD: {eod[i]:.4f}, SPD: {spd[i]:.4f}")


def main():
    pd.set_option('display.float_format', lambda x: '%.1f' % x)

    parser = argparse.ArgumentParser(description="Fair Hypernetworks")

    parser.add_argument("--data_name", type=str, default="compas", choices=["adult", "compas"], help="choice of dataset")
    parser.add_argument("--model_name", type=str, default="LR", choices=["NN", "LR"], help="choice of model")
    parser.add_argument("--num_nodes", type=int, default=4, help="number of simulated clients")
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--inner_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-6, help="weight decay")
    parser.add_argument("--gpu", type=int, default=4, help="gpu device ID")
    parser.add_argument("--fair", type=str, default="eo", choices=["none", "eo", "dp", "both"],
                        help="whether to use fairness of not.")
    parser.add_argument("--alpha", type=int, default=150, help="fairness/accuracy trade-off parameter")
    parser.add_argument("--which_position", type=int, default=5, choices=[5, 8],
                        help="which position the sensitive attribute is in. 5: compas, 8: adult")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count()
    set_logger()

    device = get_device(gpus=args.gpu)
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

