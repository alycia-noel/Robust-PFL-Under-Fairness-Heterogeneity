import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3, 4, 5, 6, 7"
import argparse
import warnings
from collections import defaultdict
import math
import torch
import numpy as np
import pandas as pd
import torch.utils.data
from tqdm import trange
from experiments.models import Plain_LR
from experiments.node import BaseNodes
from experiments.utils import seed_everything, set_logger, TP_FP_TN_FN, metrics
warnings.filterwarnings("ignore")

@torch.no_grad()
def evaluate(nodes, num_nodes, global_model, models, device, which_position, fair_eval, curr_node_id):
    results = defaultdict(lambda: defaultdict(list))
    preds = []
    true = []
    a, eod, spd = [], [], []

    for node_id in range(num_nodes):
        if fair_eval:
            node_id = curr_node_id
            model = models
        else:
            model = models[node_id]
        sd = global_model.state_dict()
        model.load_state_dict(sd)
        model.to(device)

        pred_client = []
        true_client = []
        queries_client = []


        running_loss, running_correct, running_samples = 0, 0, 0

        if fair_eval:
            curr_data = nodes.train_loaders[node_id]
        else:
            curr_data = nodes.test_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)

            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            pred = model(x)
            pred_thresh = (pred > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())

            correct = torch.eq(pred_thresh, y.unsqueeze(1)).type(torch.cuda.LongTensor)
            running_correct += torch.count_nonzero(correct).item()

            running_samples += len(y)

        tp, fp, tn, fn = TP_FP_TN_FN(queries_client, pred_client, true_client, which_position)

        accuracy, EOD, SPD = metrics(tp, fp, tn, fn)

        a.append(accuracy)
        eod.append(EOD)
        spd.append(SPD)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples
        preds.append(pred_client)
        true.append(true_client)

    total_correct = sum([val['correct'] for val in results.values()])
    total_samples = sum([val['total'] for val in results.values()])

    avg_acc = total_correct / total_samples
    all_acc = [val['correct'] / val['total'] for val in results.values()]

    if fair_eval:
        return avg_acc, eod, spd, tp, fp, tn, fn

    return avg_acc, all_acc, eod, spd

def train(device, data_name, classes_per_node, num_nodes, steps, inner_steps, inner_lr, inner_wd, bs, alpha, fair, which_position):

    models = [None for i in range(num_nodes)]
    client_fairness = []
    client_optimizers_theta = [None for i in range(num_nodes)]

    seed_everything(0)

    nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node, fairfed=True)
    num_features = len(nodes.features)

    # Note: the bound on the global model does not matter since the fairness is set to none
    global_model = Plain_LR(num_features)

    # set fairness for all clients
    if fair == 'dp':
        client_fairness = ['dp' for i in range(num_nodes)]
    elif fair == 'eo':
        client_fairness = ['eo' for i in range(num_nodes)]
    elif fair == 'both':
        for i in range(num_nodes):
            if i % 2 == 0:
                client_fairness.append('dp')
            else:
                client_fairness.append('eo')
    elif fair == 'none':
        client_fairness = ['none' for i in range(num_nodes)]


    # Set models for all clients
    for i in range(num_nodes):
        models[i] = Plain_LR(input_size=num_features)
        client_optimizers_theta[i] = torch.optim.Adam(models[i].parameters(), lr=inner_lr, weight_decay=inner_wd)

    global_model.to(device)

    loss = torch.nn.BCELoss()
    step_iter = trange(steps)

    for _ in step_iter:

        client_weights = []
        client_biases = []
        all_m_k = []
        all_F_k = []

        for j in range(num_nodes):
            node_id = j
            f_k = 0
            m_k = 0
            fair = client_fairness[j]

            # get client models and optimizers
            model = models[node_id]

            # Get F_k and m_k
            if fair == 'dp':
                avg_acc, eod, spd, tp, fp, tn, fn = evaluate(nodes=nodes, num_nodes=1, global_model=global_model, models=model, device=device, which_position=which_position, fair_eval=True, curr_node_id = node_id)

                f_k = spd[0]
                m_k = ((((tp[1] + fp[1])/ nodes.individual_stats[node_id][-1]) / nodes.individual_stats[node_id][0]) * (nodes.individual_stats[node_id][0] / nodes.stats[0])) - ((((tp[2] + fp[2])/ nodes.individual_stats[node_id][-1]) / nodes.individual_stats[node_id][1]) * (nodes.individual_stats[node_id][1] / nodes.stats[1]))
                # Pr(y^ = 1 | A = 0, C=node_id) * (Pr(A = 0 | C = node_id) / Pr(A = 0)) + Pr(y^ = 1 | A = 1, C=node_id) * (Pr(A = 1 | C = node_id) / Pr(A = 1))

            if fair == 'eo':
                avg_acc, eod, spd, tp, fp, tn, fn = evaluate(nodes=nodes, num_nodes=1, global_model=global_model, models=model, device=device, which_position=which_position, fair_eval=True, curr_node_id = node_id)
                f_k = eod[0]
                m_k = (((tp[1] / nodes.individual_stats[node_id][-1])/ nodes.individual_stats[node_id][2]) * (nodes.individual_stats[node_id][2]/nodes.stats[2])) -  (((tp[2] / nodes.individual_stats[node_id][-1])/ nodes.individual_stats[node_id][3]) * (nodes.individual_stats[node_id][3]/nodes.stats[3]))
                # Pr(y^ = 1 | A = 0, Y = 1, C = node_id) * (Pr(A = 0, Y = 1 | C = node_id) / Pr(A = 0, Y = 1)) - Pr(y^ = 1 | A = 1, Y = 1, C = node_id) * (Pr(A = 1, Y = 1 | C = node_id) / Pr(A = 1, Y = 1))

            all_F_k.append(f_k)
            all_m_k.append(m_k)

            sd = global_model.state_dict()
            model.load_state_dict(sd)
            model.to(device)
            model.train()

            inner_optim_theta = client_optimizers_theta[node_id]

            for j in range(inner_steps):
                inner_optim_theta.zero_grad()

                batch = next(iter(nodes.train_loaders[node_id]))
                x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)

                # train and update local
                pred = model(x)
                err = loss(pred, y.unsqueeze(1))
                err.backward()
                inner_optim_theta.step()

            # Store final weights of the client after local training
            client_weights.append(model.fc1.weight.data.clone())
            client_biases.append(model.fc1.bias.data.clone())

        # Update global model
        new_weights = torch.zeros(size=global_model.fc1.weight.shape)
        new_bias = torch.zeros(size=global_model.fc1.bias.shape)

        if fair != 'none':
            global_F_t = 0
            client_update_weight = []

            for x, entry in enumerate(all_m_k):
                global_F_t += ((nodes.individual_stats[x][-1] / nodes.stats[-1])*entry)

            for i in range(num_nodes):
                client_update_weight.append(math.exp(-alpha*math.fabs(all_F_k[i] - global_F_t)) * (nodes.individual_stats[i][-1] / nodes.stats[-1]))

            for i in range(num_nodes):
                new_weights += ((client_update_weight[i] / np.sum(client_update_weight)) * client_weights[i].cpu())
                new_bias += ((client_update_weight[i] / np.sum(client_update_weight)) * client_biases[i].cpu())

        else:
            for i in range(num_nodes):
                new_weights += ((nodes.individual_stats[i][-1] / nodes.stats[-1]) * client_weights[i].cpu())
                new_bias += ((nodes.individual_stats[i][-1] / nodes.stats[-1]) * client_biases[i].cpu())

        new_weights = new_weights.to(device)
        new_biases = new_bias.to(device)

        global_model.fc1.weight.data = new_weights.data.clone()
        global_model.fc1.bias.data = new_biases.data.clone()

    # Testing
    avg_acc_all, all_acc, eod, spd = evaluate(nodes=nodes, num_nodes=num_nodes, global_model=global_model, models=models, device=device, which_position=which_position, fair_eval=False, curr_node_id=None)

    print(f"\n\nFinal Results | AVG Acc: {avg_acc_all:.4f}")
    for i in range(num_nodes):
        print("\nClient", i+1)
        print(f"Acc: {all_acc[i]:.4f}, EOD: {eod[i]:.4f}, SPD: {spd[i]:.4f}")


def main():

    names = ['adult']
    fair = ['none']

    for i, n in enumerate(names):
        for j, f in enumerate(fair):
            if n == 'adult':
                important = 8
                clr = .01
                bs = 256
                alpha = 1
            elif n == 'compas':
                important = 5
                clr = .01
                bs = 64
                alpha = 20

            pd.set_option('display.float_format', lambda x: '%.1f' % x)

            parser = argparse.ArgumentParser(description="Fair Hypernetworks")

            parser.add_argument("--data_name", type=str, default=n, choices=["adult", "compas"], help="choice of dataset")
            parser.add_argument("--num_nodes", type=int, default=4, help="number of simulated clients")
            parser.add_argument("--num_steps", type=int, default=20)
            parser.add_argument("--batch_size", type=int, default=bs)
            parser.add_argument("--inner_steps", type=int, default=1, help="number of inner steps")
            parser.add_argument("--inner_lr", type=float, default=clr, help="learning rate for inner optimizer")
            parser.add_argument("--inner_wd", type=float, default=.0001, help="inner weight decay")
            parser.add_argument("--fair", type=str, default=f, choices=["none", "eo", "dp", "both"],  help="whether to use fairness of not.")
            parser.add_argument("--alpha", type=int, default=alpha, help="fairness/accuracy trade-off parameter")
            parser.add_argument("--which_position", type=int, default=important, choices=[5, 8],
                                help="which position the sensitive attribute is in. 5: compas, 8: adult")

            args = parser.parse_args()

            set_logger()

            device = "cuda:4"

            args.classes_per_node = 2

            train(
                device=device,
                data_name=args.data_name,
                classes_per_node=args.classes_per_node,
                num_nodes=args.num_nodes,
                steps=args.num_steps,
                inner_steps=args.inner_steps,
                inner_lr=args.inner_lr,
                inner_wd=args.inner_wd,
                bs=args.batch_size,
                alpha=args.alpha,
                fair=args.fair,
                which_position=args.which_position)


if __name__ == "__main__":
    main()