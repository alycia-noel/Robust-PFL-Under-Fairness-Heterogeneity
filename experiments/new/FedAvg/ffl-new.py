import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3, 4, 5, 6, 7"
import argparse
import logging
import random
import warnings
import copy
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import copy
import pandas as pd
import torch.utils.data
from tqdm import trange
from experiments.new.pFedHN.pFedHN_models import LR, Constraint
from experiments.new.pFedHN.node import BaseNodes
from experiments.new.pFedHN.utils import seed_everything, set_logger, TP_FP_TN_FN, metrics
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

def eval_model(nodes, num_nodes, hnet, model, cnet, num_features, loss, device, fair, constraint, alpha, confusion, which_position):
    curr_results, preds, true, a, f_a, m_a, eod, spd = evaluate(nodes, num_nodes, hnet, model, cnet, num_features, loss, device, fair, constraint, alpha, which_position)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    return curr_results, avg_acc, all_acc, f_a, m_a, eod, spd

@torch.no_grad()
def evaluate(nodes, num_nodes, global_model, models, cnets, num_features, loss, device, fair, constraints, alpha, which_position):
    results = defaultdict(lambda: defaultdict(list))
    preds = []
    true = []
    f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd = [], [], [], [], [], [], [], [], []

    for node_id in range(num_nodes):
        pred_client = []
        true_client = []
        queries_client = []
        model = models[node_id]
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

def train(save_file_name, device, data_name,model_name,classes_per_node,num_nodes,steps,inner_steps,lr,inner_lr,wd,inner_wd, hyper_hid,n_hidden,bs, alpha,fair, which_position):
    avg_acc = [[] for i in range(num_nodes + 1)]
    all_eod =  [[] for i in range(num_nodes)]
    all_spd = [[] for i in range(num_nodes)]
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

        total_data_length = 0
        client_data_length = []
        for i in range(num_nodes):
            total_data_length += len(nodes.train_loaders[i])
            client_data_length.append(len(nodes.train_loaders[i]))

        global_model = LR(num_features, bound=0.05, fairness='none')

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

        # Set models for all clients
        for i in range(num_nodes):
            models[i] = LR(input_size=num_features, bound=alpha[0], fairness=client_fairness[i])
            constraints[i] = Constraint(fair=client_fairness[i], bound=alpha[0])
            client_optimizers_theta[i] = torch.optim.Adam(models[i].parameters(), lr=inner_lr, weight_decay=inner_wd)
            if fair != 'none':
                client_optimizers_lambda[i] = torch.optim.Adam(constraints[i].parameters(), lr=inner_lr,
                                                               weight_decay=inner_wd)

        global_model.to(device)

        loss = torch.nn.BCELoss()
        step_iter = trange(steps)

        for step in step_iter:

            client_weights = []
            client_biases = []
            sampled = []
            choices = [0, 1, 2, 3]

            sample_precentage = random.choice([1,2,3,4])
            for j in range(sample_precentage):
                node_id = random.choice(choices)
                sampled.append(node_id)
                choices.remove(node_id)

                # get client models and optimizers
                model = models[node_id]
                if fair != 'none':
                    constraint = constraints[node_id]
                    constraint.to(device)
                alpha=alphas[node_id]

                sd = global_model.state_dict()
                model.load_state_dict(sd)
                model.to(device)
                model.train()

                inner_optim_theta = client_optimizers_theta[node_id]
                inner_optim_lambda = client_optimizers_lambda[node_id]

                for j in range(inner_steps):
                    inner_optim_theta.zero_grad()
                    if fair != 'none':
                        inner_optim_lambda.zero_grad()

                    batch = next(iter(nodes.train_loaders[node_id]))
                    x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
                    s = x[:, which_position].to(device)

                    # train and update local
                    pred, m_mu_q = model(x, s, y)

                    if fair == 'none':
                        err = loss(pred, y.unsqueeze(1))
                    else:
                        err = loss(pred, y.unsqueeze(1)) + constraint(m_mu_q)

                    err.backward()

                    if fair != 'none':
                        for group in inner_optim_lambda.param_groups:
                            for p in group['params']:
                                p.grad = -1 * p.grad

                    inner_optim_theta.step()

                    if fair != 'none':
                        inner_optim_lambda.step()

                # delta theta and global updates
                client_weights.append(model.fc1.weight.data.clone())
                client_biases.append(model.fc1.bias.data.clone())

            new_weights = torch.zeros(size=global_model.fc1.weight.shape)
            new_bias = torch.zeros(size=global_model.fc1.bias.shape)

            total_sampled_length = 0
            for i, c in enumerate(sampled):
                total_sampled_length += client_data_length[c]
            for i, c in enumerate(sampled):
                new_weights += ((client_data_length[c] / total_sampled_length) * client_weights[i].cpu())
                new_bias += ((client_data_length[c] / total_sampled_length) * client_biases[i].cpu())
            new_weights = (new_weights).to(device)
            new_biases = (new_bias).to(device)
            global_model.fc1.weight.data = new_weights.data.clone()
            global_model.fc1.bias.data = new_biases.data.clone()

        step_results, avg_acc_all, all_acc, f_a, m_a, eod, spd = eval_model(
            nodes, num_nodes, global_model, models, None, num_features, loss, device, confusion=False, fair=fair,
            constraint=constraints, alpha=alpha, which_position=which_position)

        logging.info(f"\n\nFinal Results | AVG Acc: {avg_acc_all:.4f}")
        avg_acc[0].append(avg_acc_all)
        for i in range(num_nodes):
            avg_acc[i + 1].append(all_acc[i])
            all_eod[i].append(eod[i])
            all_spd[i].append(spd[i])

    # file = open(save_file_name, "a")
    # file.write(
    #     "\n AVG Acc: {0:.4f}, C1 ACC: {1:.4f}, C2 ACC: {2:.4f}, C3 ACC: {3:.4f}, C4 ACC: {4:.4f}, C1 F1: {5:.4f}, C2 F1: {6:.4f}, C3 F1: {7:.4f}, C4 F1: {8:.4f}, C1 AOD: {9:.4f}, C2 AOD: {10: .4f}, C3 AOD: {11:.4f}, C4 AOD: {12:.4f}, C1 EOD: {13: .4f}, C2 EOD: {14:.4f}, C3 EOD: {15:.4f}, C4 EOD: {16:.4f}, C1 SPD: {17:.4f}, C2 SPD: {18:.4f}, C3 SPD: {19:.4f}, C4 SPD: {20:.4f}, Time: {21:.2f}".format(np.mean(avg_acc[0]),np.mean(avg_acc[1]), np.mean(avg_acc[2]), np.mean(avg_acc[3]), np.mean(avg_acc[4]), np.mean(all_f1[0]), np.mean(all_f1[1]), np.mean(all_f1[2]), np.mean(all_f1[3]), np.mean(all_aod[0]), np.mean(all_aod[1]), np.mean(all_aod[2]), np.mean(all_aod[3]), np.mean(all_eod[0]), np.mean(all_eod[1]), np.mean(all_eod[2]), np.mean(all_eod[3]), np.mean(all_spd[0]), np.mean(all_spd[1]), np.mean(all_spd[2]), np.mean(all_spd[3]), step_iter.format_dict['elapsed']))
    # file.close()

    print(f"\n\nFinal Results | AVG Acc: {np.mean(avg_acc[0]):.4f}")
    for i in range(num_nodes):
        print("\nClient", i+1)
        print(f"Acc: {np.mean(avg_acc[i+1]):.4f}, EOD: {np.mean(all_eod[i]):.4f}, SPD: {np.mean(all_spd[i]):.4f}")


def main():
    # file = open("/home/ancarey/FairFLHN/experiments/new/FedAvg/all-runs.txt", "w")
    # file.close()

    names = ['compas']#, 'compas']
    fair = ['dp']#['dp', 'eo', 'both']

    for i, n in enumerate(names):
        for j, f in enumerate(fair):
            if n == 'adult':
                important = 8
                clr = .001
                hlr = 1e-5
                bs = 256
                a1 = .001
                a2 = 100
            elif n == 'compas':
                important = 5
                clr = .05
                hlr = 5e-5
                bs = 64
                a1 = .01
                a2 = 40

            print(clr, hlr, a1, n, f)
            pd.set_option('display.float_format', lambda x: '%.1f' % x)

            writer = SummaryWriter('results')

            parser = argparse.ArgumentParser(description="Fair Hypernetworks")

            parser.add_argument("--data_name", type=str, default=n, choices=["adult", "compas"], help="choice of dataset")
            parser.add_argument("--model_name", type=str, default="LR", choices=["NN", "LR"], help="choice of model")
            parser.add_argument("--num_nodes", type=int, default=4, help="number of simulated clients")
            parser.add_argument("--num_steps", type=int, default=5000)
            parser.add_argument("--batch_size", type=int, default=bs)
            parser.add_argument("--inner_steps", type=int, default=50, help="number of inner steps")
            parser.add_argument("--n_hidden", type=int, default=4, help="num. hidden layers")
            parser.add_argument("--inner_lr", type=float, default=clr, help="learning rate for inner optimizer")
            parser.add_argument("--lr", type=float, default=hlr, help="learning rate")
            parser.add_argument("--wd", type=float, default=1e-10, help="weight decay")
            parser.add_argument("--inner_wd", type=float, default=1e-10, help="inner weight decay")
            parser.add_argument("--embed_dim", type=int, default=10, help="embedding dim")
            parser.add_argument("--hyper_hid", type=int, default=100, help="hypernet hidden dim")
            parser.add_argument("--gpu", type=int, default=5, help="gpu device ID")
            parser.add_argument("--eval_every", type=int, default=50, help="eval every X selected epochs")
            parser.add_argument("--save_path", type=str, default="/home/ancarey/FairFLHN/experiments/adult/results",
                                help="dir path for output file")
            parser.add_argument("--seed", type=int, default=0, help="seed value")
            parser.add_argument("--fair", type=str, default=f, choices=["none", "eo", "dp", "both"],
                                help="whether to use fairness of not.")
            parser.add_argument("--alpha", type=int, default=[a1,a2], help="fairness/accuracy trade-off parameter")
            parser.add_argument("--which_position", type=int, default=important, choices=[5, 8],
                                help="which position the sensitive attribute is in. 5: compas, 8: adult")
            parser.add_argument("--save_file_name", type=str,
                                default="/home/ancarey/FairFLHN/experiments/new/FedAvg/all-runs.txt")

            args = parser.parse_args()
            assert args.gpu <= torch.cuda.device_count()
            set_logger()

            device = "cuda:4"

            args.classes_per_node = 2

            train(
                save_file_name=args.save_file_name,
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
                which_position=args.which_position)


if __name__ == "__main__":
    main()