import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3, 4, 5, 6, 7"
import argparse
import logging
import random
import warnings
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import pandas as pd
import torch.utils.data
from tqdm import trange
from pFedHN_models import LRHyper, LR, Constraint
from node import BaseNodes
from utils import seed_everything, set_logger, TP_FP_TN_FN, metrics
warnings.filterwarnings("ignore")

def eval_model(nodes, num_nodes, hnet, model, device, which_position):
    curr_results, preds, true, a, f_a, m_a, eod, spd = evaluate(nodes, num_nodes, hnet, model, device, which_position)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    return curr_results, avg_acc, all_acc, f_a, m_a, eod, spd

@torch.no_grad()
def evaluate(nodes, num_nodes, hnet, models, device, which_position):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))
    preds = []
    true = []
    a, f_a, m_a, eod, spd = [], [], [], [], []

    for node_id in range(num_nodes):
        pred_client = []
        true_client = []
        queries_client = []
        model = models[node_id]
        model.eval()
        model.to(device)

        running_loss, running_correct, running_samples = 0, 0, 0

        curr_data = nodes.test_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
            s = x[:, which_position].to(device)

            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
            model.load_state_dict(weights)

            pred, m_mu_q = model(x, s, y)
            pred_thresh = (pred > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())

            correct = torch.eq(pred_thresh, y.unsqueeze(1)).type(torch.cuda.LongTensor)
            running_correct += torch.count_nonzero(correct).item()

            running_samples += len(y)

        tp, fp, tn, fn = TP_FP_TN_FN(queries_client, pred_client, true_client, which_position)

        accuracy, f_acc, m_acc, EOD, SPD = metrics(tp, fp, tn, fn)

        a.append(accuracy)
        f_a.append(f_acc)
        m_a.append(m_acc)
        eod.append(EOD)
        spd.append(SPD)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples
        preds.append(pred_client)
        true.append(true_client)

    return results, preds, true, a, f_a, m_a, eod, spd

def train(device, data_name, classes_per_node, num_nodes, steps, inner_steps, lr, inner_lr, wd, inner_wd, hyper_hid, n_hidden, bs, alpha, fair, which_position):

    avg_acc = [[] for i in range(num_nodes + 1)]
    all_eod =  [[] for i in range(num_nodes)]
    all_spd = [[] for i in range(num_nodes)]
    all_times = [[] for i in range(10)]
    models = [None for i in range(num_nodes)]
    constraints = [None for i in range(num_nodes)]
    client_fairness = []
    client_optimizers_theta = [None for i in range(num_nodes)]
    client_optimizers_lambda = [None for i in range(num_nodes)]
    alphas = []

    for i in range(1):
        seed_everything(0)

        nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node)
        num_features = len(nodes.features)
        embed_dim = num_features

        if fair == 'dp':
            client_fairness = ['dp' for i in range(num_nodes)]
            alphas = [alpha[0] for i in range(num_nodes)]
        elif fair == 'eo':
            client_fairness = ['eo' for i in range(num_nodes)]
            alphas = [alpha[1] for i in range(num_nodes)]
        elif fair == 'both':
            for i in range(num_nodes):
                if i % 2 == 0:
                    client_fairness.append('dp')
                    alphas.append(alpha[0])
                else:
                    client_fairness.append('eo')
                    alphas.append(alpha[1])
        elif fair == 'none':
            client_fairness = ['none' for i in range(num_nodes)]
            alphas = ['none' for i in range(num_nodes)]

        hnet = LRHyper(device=device, n_nodes=num_nodes, embedding_dim=embed_dim, context_vector_size=num_features,
                       hidden_size=num_features, hnet_hidden_dim=hyper_hid, hnet_n_hidden=n_hidden)

        for i in range(num_nodes):
            models[i] = LR(input_size=num_features, bound=alpha[0], fairness=client_fairness[i])
            constraints[i] = Constraint(fair=client_fairness[i], bound=alpha[0])
            client_optimizers_theta[i] = torch.optim.Adam(models[i].parameters(), lr=inner_lr, weight_decay=inner_wd)
            if fair != 'none':
                client_optimizers_lambda[i] = torch.optim.Adam(constraints[i].parameters(), lr=inner_lr, weight_decay=inner_wd)

        hnet.to(device)

        optimizer = torch.optim.Adam(params=hnet.parameters(), lr=lr, weight_decay=wd)
        loss = torch.nn.BCELoss()
        step_iter = trange(steps)

        nodes_choices = [0, 1, 2, 3]

        for _ in step_iter:
            hnet.train()
            node_id = random.choice(nodes_choices)

            model = models[node_id]

            alpha=alphas[node_id]

            if fair != 'none':
                constraint = constraints[node_id]
                constraint.to(device)

            inner_optim_theta = client_optimizers_theta[node_id]
            inner_optim_lambda = client_optimizers_lambda[node_id]

            weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
            model.load_state_dict(weights)
            model.to(device)

            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

            model.train()
            for j in range(inner_steps):
                inner_optim_theta.zero_grad()
                if fair != 'none':
                    inner_optim_lambda.zero_grad()
                optimizer.zero_grad()

                batch = next(iter(nodes.train_loaders[node_id]))
                x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
                s = x[:, which_position].to(device)

                pred, m_mu_q = model(x, s, y)

                if fair == 'none':
                    err = loss(pred, y.unsqueeze(1))
                else:
                    err = loss(pred, y.unsqueeze(1)) + constraint(m_mu_q)

                err.backward()

                if fair != 'none':
                    for group in inner_optim_lambda.param_groups:
                        for p in group['params']:
                            p.grad = -1*p.grad
                            # if node_id == 3:
                            #     print(p.grad)

                inner_optim_theta.step()

                if fair != 'none':
                    inner_optim_lambda.step()

            optimizer.zero_grad()
            final_state = model.state_dict()
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
            hnet_grads = torch.autograd.grad(list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()))

            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g

            optimizer.step()

        step_results, avg_acc_all, all_acc, f_a, m_a, eod, spd = eval_model(nodes=nodes, num_nodes=num_nodes, hnet=hnet, model=models, device=device, which_position=which_position)
        logging.info(f"\n\nFinal Results | AVG Acc: {avg_acc_all:.4f}")
        avg_acc[0].append(avg_acc_all)
        for i in range(num_nodes):
            avg_acc[i + 1].append(all_acc[i])
            all_eod[i].append(eod[i])
            all_spd[i].append(spd[i])
        all_times.append(step_iter.format_dict["elapsed"])


    file = open("/home/ancarey/FairFLHN/experiments/new/pFedHN/results/four-layer-hp-compas.txt", "a")
    file.write(
        "\n CLR: {13}, HLR: {14}, AVG Acc: {0:.4f}, C1 ACC: {1:.4f}, C2 ACC: {2:.4f}, C3 ACC: {3:.4f}, C4 ACC: {4:.4f}, C1 EOD: {5: .4f}, C2 EOD: {6:.4f}, C3 EOD: {7:.4f}, C4 EOD: {8:.4f}, C1 SPD: {9:.4f}, C2 SPD: {10:.4f}, C3 SPD: {11:.4f}, C4 SPD: {12:.4f}".format(np.mean(avg_acc[0]),np.mean(avg_acc[1]), np.mean(avg_acc[2]), np.mean(avg_acc[3]), np.mean(avg_acc[4]), np.mean(all_eod[0]), np.mean(all_eod[1]), np.mean(all_eod[2]), np.mean(all_eod[3]), np.mean(all_spd[0]), np.mean(all_spd[1]), np.mean(all_spd[2]), np.mean(all_spd[3]), inner_lr, lr ))
    file.close()

    print(f"\n\nFinal Results | AVG Acc: {np.mean(avg_acc[0]):.4f}")
    print(all_spd)
    for i in range(num_nodes):
        print("\nClient", i+1)
        print(f"Acc: {np.mean(avg_acc[i+1]):.4f}, EOD: {np.mean(all_eod[i]):.4f}, SPD: {np.mean(all_spd[i]):.4f}")


def main():
    file = open("/home/ancarey/FairFLHN/experiments/new/pFedHN/results/four-layer-hp-adult.txt", "w")
    file.close()

    #dp = [.005]#.0000001, .5, .0025]#[.009, .008, .007, .006, .005, .004, .003, .002, .001, .0009, .0008, .0007, .0006] #.005, .0005 best
    dp = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    hlr = [1e-5, 5e-5, 1e-4, 5e-4,]
    steps = [5000]#[500, 1000, 2000, 2500, 5000]

    for i, d in enumerate(dp):
        for j, h in enumerate(hlr):
            for k, s in enumerate(steps):
                print('clr: ', d, 'hlr: ', h)
                pd.set_option('display.float_format', lambda x: '%.1f' % x)

                parser = argparse.ArgumentParser(description="Fair Hypernetworks")

                parser.add_argument("--data_name", type=str, default="adult", choices=["adult", "compas"], help="choice of dataset")
                parser.add_argument("--num_nodes", type=int, default=4, help="number of simulated clients")
                parser.add_argument("--num_steps", type=int, default=s)
                parser.add_argument("--batch_size", type=int, default=256)
                parser.add_argument("--inner_steps", type=int, default=50, help="number of inner steps")
                parser.add_argument("--n_hidden", type=int, default=4, help="num. hidden layers")
                parser.add_argument("--inner_lr", type=float, default=d, help="learning rate for inner optimizer")
                parser.add_argument("--lr", type=float, default=h, help="learning rate")
                parser.add_argument("--wd", type=float, default=1e-10, help="weight decay")
                parser.add_argument("--inner_wd", type=float, default=1e-10, help="inner weight decay")
                parser.add_argument("--hyper_hid", type=int, default=100, help="hypernet hidden dim")
                parser.add_argument("--seed", type=int, default=0, help="seed value")
                parser.add_argument("--fair", type=str, default="none", choices=["none", "eo", "dp", "both"],
                                    help="whether to use fairness of not.")
                parser.add_argument("--alpha", type=int, default=[d,100], help="fairness/accuracy trade-off parameter")
                parser.add_argument("--which_position", type=int, default=8, choices=[5, 8],
                                    help="which position the sensitive attribute is in. 5: compas, 8: adult")
                args = parser.parse_args()
                set_logger()
                device = "cuda:5"

                args.classes_per_node = 2

                train(
                    device=device,
                    data_name=args.data_name,
                    classes_per_node=args.classes_per_node,
                    num_nodes=args.num_nodes,
                    steps=args.num_steps,
                    inner_steps=args.inner_steps,
                    lr=args.lr,
                    inner_lr=args.inner_lr,
                    wd=args.wd,
                    inner_wd=args.inner_wd,
                    hyper_hid=args.hyper_hid,
                    n_hidden=args.n_hidden,
                    bs=args.batch_size,
                    alpha=args.alpha,
                    fair=args.fair,
                    which_position=args.which_position)


if __name__ == "__main__":
    main()