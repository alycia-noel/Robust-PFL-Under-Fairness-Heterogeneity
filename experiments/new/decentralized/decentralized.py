import time
import argparse
import warnings
import torch
import torch.nn as nn
from collections import defaultdict
from experiments.new.utils import seed_everything, metrics, TP_FP_TN_FN, set_logger, get_device
from experiments.new.models import LR_plain
from experiments.new.node import BaseNodes
from fairtorch import DemographicParityLoss, EqualiedOddsLoss
warnings.filterwarnings("ignore")
import pandas as pd

@torch.no_grad()
def evaluate(model_dict, name_of_models, criterion_dict, name_of_criterions, fair, fair_loss_dict, name_of_fair_loss, nodes, which_position):

    results = defaultdict(lambda: defaultdict(list))

    f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd = [], [], [], [], [], [], [], [], []

    for node_id in range(len(nodes)):
        model = model_dict[name_of_models[node_id]]
        model.eval()
        loss = criterion_dict[name_of_criterions[node_id]]
        if fair != "none":
            fair_loss = fair_loss_dict[name_of_fair_loss[node_id]]

        pred_client = []
        true_client = []
        queries_client = []

        running_loss, running_correct, running_samples = 0, 0, 0

        curr_data = nodes.test_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.FloatTensor)) for t in batch)
            s = x[:, which_position]
            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            pred = model(x)
            pred_prob = torch.sigmoid(pred)
            pred_thresh = (pred_prob > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())

            if fair == 'none':
                running_loss += loss(pred, y.unsqueeze(1)).item()
            else:
                running_loss += (loss(pred, y.unsqueeze(1)).item() + fair_loss(x, pred, s, y).item())

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

    all_acc = [val['correct'] / val['total'] for val in results.values()]
    all_loss = [val['loss'] for val in results.values()]

    return results, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd


def create_model_optimizer_criterion_dict(number_of_samples, fair, alpha, m, learning_rate, wd, num_features):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()
    fair_loss_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = LR_plain(input_size=num_features)

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
        elif fair == "both":
            fair_loss_name = "fair_loss" + str(i)
            if i % 2 == 0:
                fair_loss_info = DemographicParityLoss(sensitive_classes=[0, 1], alpha=100)
            else:
                fair_loss_info = EqualiedOddsLoss(sensitive_classes=[0, 1], alpha=250)
            fair_loss_dict.update({fair_loss_name: fair_loss_info})

    return model_dict, optimizer_dict, criterion_dict, fair_loss_dict


def start_train_end_node_process_print_some(client, model_dict, name_of_models, criterion_dict, name_of_criterions, optimizer_dict, name_of_optimizers, fair_loss_dict, name_of_fair_loss, num_epoch, nodes, fair, which_position):
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
            fair_l = fair_loss(x, pred, s, y)
            err = criterion(pred, y.unsqueeze(1)) + fair_l

        err.backward()
        optimizer.step()


def train(data_name,model_name,classes_per_node,num_nodes,lr,wd,bs, alpha,fair, which_position, inner_steps):
    seed_everything(0)

    nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node)

    num_features = len(nodes.features)

    times_all, roc_all = [], []

    model_dict, optimizer_dict, criterion_dict, fair_loss_dict = create_model_optimizer_criterion_dict(num_nodes, fair, alpha, model_name, lr, wd, num_features)

    name_of_models = list(model_dict.keys())
    name_of_optimizers = list(optimizer_dict.keys())
    name_of_criterions = list(criterion_dict.keys())
    name_of_fair_loss = list(fair_loss_dict.keys())

    print('~' * 22)
    print('~~~ Start Training ~~~')
    print('~' * 22, '\n')

    for client in range(num_nodes):
        start = time.time()
        start_train_end_node_process_print_some(client, model_dict, name_of_models, criterion_dict, name_of_criterions, optimizer_dict, name_of_optimizers, fair_loss_dict, name_of_fair_loss, inner_steps, nodes, fair, which_position)
        end = time.time()

        times_all.append(end - start)

    results, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd = evaluate(model_dict, name_of_models, criterion_dict, name_of_criterions, fair, fair_loss_dict, name_of_fair_loss, nodes, which_position)
    for i in range(num_nodes):
        print("\nClient", i + 1)
        print(f"Acc: {all_acc[i]:.4f}, F Acc: {f_a[i]:.4f}, M Acc: {m_a[i]:.4f}, F1: {f1[i]:.4f}, AOD: {aod[i]:.4f}, EOD: {eod[i]:.4f}, SPD: {spd[i]:.4f}")
    print(sum(times_all))

def main():
    pd.set_option('display.float_format', lambda x: '%.1f' % x)

    parser = argparse.ArgumentParser(description="Fair Hypernetworks")

    parser.add_argument("--data_name", type=str, default="adult", choices=["adult", "compas"], help="choice of dataset")
    parser.add_argument("--model_name", type=str, default="LR", choices=["NN", "LR"], help="choice of model")
    parser.add_argument("--num_nodes", type=int, default=4, help="number of simulated clients")
    parser.add_argument("--inner_steps", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--gpu", type=int, default=4, help="gpu device ID")
    parser.add_argument("--fair", type=str, default="eo", choices=["none", "eo", "dp", "both"],
                        help="whether to use fairness of not.")
    parser.add_argument("--alpha", type=int, default=250, help="fairness/accuracy trade-off parameter")
    parser.add_argument("--which_position", type=int, default=5, choices=[5, 8],
                        help="which position the sensitive attribute is in. 5: compas, 8: adult")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count()
    set_logger()

    device = get_device(gpus=args.gpu)
    args.class_per_node = 2

    train(
        data_name=args.data_name,
        model_name = args.model_name,
        classes_per_node = args.class_per_node,
        num_nodes=args.num_nodes,
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

