import os
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
from models import LR, Context, LRHyper, Constraint
from node import BaseNodes
from utils import seed_everything, set_logger, TP_FP_TN_FN, metrics, make_ascent
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sn
warnings.filterwarnings("ignore")

def eval_model(nodes, num_nodes, hnet, model, cnet, num_features, loss, device, fair, constraint, alpha, confusion, which_position):
    curr_results, pred, true, f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd = evaluate(nodes, num_nodes, hnet, model, cnet, num_features, loss, device, fair, constraint, alpha, which_position)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])

    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]
    all_loss = [val['loss'] for val in curr_results.values()]

    if confusion:

        for i in range(len(pred)):
            actual = pd.Series(true[i], name='Actual')
            prediction = pd.Series(pred[i], name='Predicted')
            confusion = pd.crosstab(actual, prediction)
            print(confusion)
            plt.figure(figsize=(12, 7))
            sn.heatmap(confusion, annot=True)
            title = 'Confusion Matrix for Client ' + str(i + 1)
            plt.title(title)
            plt.show()

    return curr_results, avg_loss, avg_acc, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd

@torch.no_grad()
def evaluate(nodes, num_nodes, hnet, models, cnets, num_features, loss, device, fair, constraints, alpha, which_position):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))
    preds = []
    true = []
    f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd = [], [], [], [], [], [], [], [], []

    for node_id in range(num_nodes):
        pred_client = []
        true_client = []
        queries_client = []
        model = models[node_id]
        cnet = cnets[node_id]
        constraint = constraints[node_id]

        model.to(device)
        cnet.to(device)
        constraint.to(device)

        running_loss, running_correct, running_samples = 0, 0, 0

        curr_data = nodes.test_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
            s = x[:, which_position].to(device)

            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            avg_context_vector, prediction_vector = cnet(x, num_features)
            weights = hnet(avg_context_vector, torch.tensor([node_id], dtype=torch.long).to(device))
            model.load_state_dict(weights)

            pred, m_mu_q = model(prediction_vector, s, y) # y is only passed to calculate m_mu_q

            pred_thresh = (pred > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())

            if fair == 'none':
                running_loss += loss(pred, y.unsqueeze(1)).item()
            else:
                running_loss += ((loss(pred, y.unsqueeze(1)) + alpha*constraint(m_mu_q).to(device)).item()) / len(batch)

            correct = torch.eq(pred_thresh,y.unsqueeze(1)).type(torch.cuda.LongTensor)
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
        results[node_id]['loss'] = running_loss
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples
        preds.append(pred_client)
        true.append(true_client)

    return results, preds, true, f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd

def train(writer, device, data_name,model_name,classes_per_node,num_nodes,steps,inner_steps,lr,inner_lr,wd,inner_wd, hyper_hid,n_hidden,bs, alpha,fair, which_position, context_hidden_size, save_file_name):
    avg_acc = [[] for i in range(num_nodes + 1)]
    all_f1 = [[] for i in range(num_nodes)]
    all_aod = [[] for i in range(num_nodes)]
    all_eod = [[] for i in range(num_nodes)]
    all_spd = [[] for i in range(num_nodes)]
    all_times = []
    models = []
    cnets = []
    constraints = []
    client_fairness = []
    client_optimizers = []
    combo_parameters = []
    alphas = []

    for i in range(1):
        seed_everything(0)

        nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node)
        num_features = len(nodes.features)
        embed_dim = num_features

        # set fairness for all clients
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

        # Set models for all clients
        for i in range(num_nodes):
            models.append(LR(input_size=num_features, bound=0.05, fairness=client_fairness[i]))
            cnets.append(Context(input_size=num_features, context_vector_size=num_features, context_hidden_size=context_hidden_size))
            constraints.append(Constraint(fair=client_fairness[i]))
            #constraints.append(Constraint())
            if fair == 'none':
                combo_parameters.append(list(models[i].parameters()) + list(cnets[i].parameters()))
            else:
                combo_parameters.append(list(models[i].parameters()) + list(cnets[i].parameters()) + list(constraints[i].parameters()))
            client_optimizers.append(torch.optim.Adam(combo_parameters[i], lr=inner_lr, weight_decay=inner_wd))

        hnet.to(device)

        optimizer = torch.optim.Adam(params=hnet.parameters(), lr=lr, weight_decay=wd)
        loss = torch.nn.BCELoss()
        step_iter = trange(steps)

        for step in step_iter:
            hnet.train()
            node_id = random.choice(range(num_nodes))

            # get client models and optimizers
            model = models[node_id]
            cnet = cnets[node_id]
            constraint = constraints[node_id]
            combo_params = combo_parameters[node_id]
            alpha = alphas[node_id]
            model.to(device)
            cnet.to(device)
            constraint.to(device)

            inner_optim = client_optimizers[node_id]

            node_c_i = nodes.c_i[node_id]

            weights = hnet(node_c_i, torch.tensor([node_id], dtype=torch.long).to(device))
            model.load_state_dict(weights)

            inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

            avg_c_i = []


            for j in range(inner_steps):
                model.train()

                inner_optim.zero_grad()
                optimizer.zero_grad()

                batch = next(iter(nodes.train_loaders[node_id]))
                x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
                s = x[:,which_position].to(device)

                avg_context_vector, pred_vec = cnet(x, num_features)
                pred, m_mu_q = model(pred_vec, s, y) # we pass y only for m_mu_q calculation

                avg_c_i.append(avg_context_vector)

                if fair == 'none':
                    err = loss(pred, y.unsqueeze(1))
                else:
                    err = loss(pred, y.unsqueeze(1)) + alpha*constraint(m_mu_q)

                err.backward()

                if fair != 'none':
                    combo_params[len(combo_params) - 1].grad.data = -1 * combo_params[len(combo_params) - 1].grad.data

                inner_optim.step()

            nodes.c_i[node_id] = torch.cuda.FloatTensor([sum(sub_list) / len(sub_list) for sub_list in zip(*avg_c_i)])

            optimizer.zero_grad()
            final_state = model.state_dict()
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
            hnet_grads = torch.autograd.grad(list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()))

            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g

            optimizer.step()

            # if step % 99 == 0 or step == 1999 or step == 0:
            #     step_results, avg_loss, avg_acc_all, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd = eval_model(nodes, num_nodes, hnet, models, cnets, num_features, loss, device, confusion=False, fair=fair, constraint=constraints, alpha=alpha, which_position=which_position)
            #
            #     logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc_all:.4f}")

        step_results, avg_loss, avg_acc_all, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd = eval_model(nodes, num_nodes, hnet, models, cnets, num_features, loss, device, confusion=False,fair=fair, constraint=constraints, alpha=alpha, which_position=which_position)
        # logging.info(f"\n\nFinal Results | AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc_all:.4f}")
        avg_acc[0].append(avg_acc_all)
        for i in range(num_nodes):
            avg_acc[i + 1].append(all_acc[i])
            all_f1[i].append(f1[i])
            all_aod[i].append(aod[i])
            all_eod[i].append(eod[i])
            all_spd[i].append(spd[i])
        all_times.append(step_iter.format_dict["elapsed"])

    file = open(save_file_name, "a")
    file.write("\nLR: {0}, HN LR: {1}, WD: {2}, AVG Acc: {3:.4f}".format(inner_lr, lr, wd, np.mean(avg_acc[0])))
    file.close()


def main():
    file = open("/home/ancarey/FairFLHN/experiments/new/grid_search_adult/results/3_32_50.txt", "w")
    file.close()

    client_lr = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    hn_lr = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    wd = [1e-10, 1e-8, 1e-6]

    for i, clr in enumerate(client_lr):
        for j, hnlr in enumerate(hn_lr):
            for k, w in enumerate(wd):

                pd.set_option('display.float_format', lambda x: '%.1f' % x)

                writer = SummaryWriter('results')

                parser = argparse.ArgumentParser(description="Fair Hypernetworks")

                parser.add_argument("--data_name", type=str, default="adult", choices=["adult", "compas"], help="choice of dataset")
                parser.add_argument("--model_name", type=str, default="LR", choices=["NN", "LR"], help="choice of model")
                parser.add_argument("--num_nodes", type=int, default=4, help="number of simulated clients")
                parser.add_argument("--num_steps", type=int, default=2000)
                parser.add_argument("--batch_size", type=int, default=32)
                parser.add_argument("--inner_steps", type=int, default=50, help="number of inner steps")
                parser.add_argument("--n_hidden", type=int, default=3, help="num. hidden layers")
                parser.add_argument("--inner_lr", type=float, default=clr, help="learning rate for inner optimizer")
                parser.add_argument("--lr", type=float, default=hnlr, help="learning rate")
                parser.add_argument("--wd", type=float, default=w, help="weight decay")
                parser.add_argument("--inner_wd", type=float, default=w, help="inner weight decay")
                parser.add_argument("--embed_dim", type=int, default=10, help="embedding dim")
                parser.add_argument("--hyper_hid", type=int, default=100, help="hypernet hidden dim")
                parser.add_argument("--gpu", type=int, default=4, help="gpu device ID")
                parser.add_argument("--eval_every", type=int, default=50, help="eval every X selected epochs")
                parser.add_argument("--save_path", type=str, default="/home/ancarey/FairFLHN/experiments/adult/results",
                                    help="dir path for output file")
                parser.add_argument("--seed", type=int, default=0, help="seed value")
                parser.add_argument("--fair", type=str, default="none", choices=["none", "eo", "dp", "both"],
                                    help="whether to use fairness of not.")
                parser.add_argument("--alpha", type=int, default=[125, 50], help="fairness/accuracy trade-off parameter")
                parser.add_argument("--which_position", type=int, default=8, choices=[5, 8],
                                    help="which position the sensitive attribute is in. 5: compas, 8: adult")
                parser.add_argument("--context_hidden_size", type=int, default=50, choices=[25, 50, 100],
                                    help="size of hidden layers of context network")
                parser.add_argument("--save_file_name", type=str, default="/home/ancarey/FairFLHN/experiments/new/grid_search_adult/results/3_32_50.txt")

                args = parser.parse_args()
                set_logger()

                device = "cuda:4"

                args.classes_per_node = 2

                train(
                    writer,
                    device=device,
                    data_name=args.data_name,
                    model_name=args.model_name,
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
                    which_position=args.which_position,
                    context_hidden_size=args.context_hidden_size,
                    save_file_name=args.save_file_name)

if __name__ == "__main__":
    main()