import os
import seaborn as sns
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3, 4, 5, 6, 7"
import argparse
import random
import warnings
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import pandas as pd
import torch.utils.data
from tqdm import trange
from models import LRHyper, LR, Constraint
from node import BaseNodes
from utils import seed_everything, set_logger, TP_FP_TN_FN, metrics
warnings.filterwarnings("ignore")

def eval_model(nodes, num_nodes, hnet, model, device, which_position):
    curr_results, preds, true, a, eod, spd = evaluate(nodes, num_nodes, hnet, model, device, which_position)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    return curr_results, avg_acc, all_acc, eod, spd

@torch.no_grad()
def evaluate(nodes, num_nodes, hnet, models, device, which_position):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))
    preds = []
    true = []
    a, eod, spd = [], [], []

    node_ids = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

    for node_id in range(num_nodes):
        pred_client = []
        true_client = []
        queries_client = []

        if num_nodes == 10:
            node_id_current = node_ids[node_id]
        else:
            node_id_current = node_id

        model = models[node_id_current]
        model.eval()
        model.to(device)

        running_loss, running_correct, running_samples = 0, 0, 0

        curr_data = nodes[node_id]

        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
            s = x[:, which_position].to(device)

            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            weights = hnet(torch.tensor([node_id_current], dtype=torch.long).to(device))
            model.load_state_dict(weights)

            pred, m_mu_q = model(x, s, y)
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

    return results, preds, true, a, eod, spd

def train(device, data_name, classes_per_node, num_nodes, steps, inner_steps, lr, inner_lr, wd, inner_wd, hyper_hid, n_hidden, bs, alpha_all, fair, which_position):

    repeated_acc = []
    repeated_spd = []
    repeated_eod = []

    tv_f = []

    b = 1/alpha_all[0]

    ranges = [2, 4, 6, 8, 10]
    for i,r in enumerate(ranges):
        all_acc_10 = []
        all_spd_10 = []
        all_eod_10 = []
        avg_acc_difference_all = []
        avg_eod_difference_all = []
        avg_spd_difference_all = []
        TV = []

        for n in range(1):
            seed_everything(0)
            print('\n\nRound', n, 'for alpha', r)
            print('======================')
            models = [None for i in range(num_nodes)]
            constraints = [None for i in range(num_nodes)]
            client_fairness = []
            client_optimizers_theta = [None for i in range(num_nodes)]
            client_optimizers_lambda = [None for i in range(num_nodes)]
            alphas = []

            c_acc_p_epoch = []
            acc_p_epoch = []

            c_eod_p_epoch = []
            eod_p_epoch = []

            c_spd_p_epoch = []
            spd_p_epoch = []

            nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node, False, r)

            training_nodes_train = nodes.train_loaders[0:90]
            training_nodes_test = nodes.test_loaders[0:90]

            novel_nodes_train = nodes.train_loaders[90:100]
            novel_nodes_test = nodes.test_loaders[90:100]

            num_features = len(nodes.features)
            embed_dim = num_features

            if fair == 'dp':
                client_fairness = ['dp' for i in range(num_nodes)]
                alphas = [alpha_all[0] for i in range(num_nodes)]
            elif fair == 'eo':
                client_fairness = ['eo' for i in range(num_nodes)]
                alphas = [alpha_all[1] for i in range(num_nodes)]
            elif fair == 'both':
                for i in range(num_nodes):
                    if i % 2 == 0:
                        client_fairness.append('dp')
                        alphas.append(alpha_all[0])
                    else:
                        client_fairness.append('eo')
                        alphas.append(alpha_all[1])
            elif fair == 'none':
                client_fairness = ['none' for i in range(num_nodes)]
                alphas = [1 for i in range(num_nodes)]

            hnet = LRHyper(device=device, n_nodes=num_nodes-10, embedding_dim=embed_dim, context_vector_size=num_features,
                           hidden_size=num_features, hnet_hidden_dim=hyper_hid, hnet_n_hidden=n_hidden)

            for i in range(num_nodes):
                models[i] = LR(input_size=num_features, bound=alphas[i], fairness=client_fairness[i])
                constraints[i] = Constraint(bound=alphas[i], fair=client_fairness[i])
                client_optimizers_theta[i] = torch.optim.Adam(models[i].parameters(), lr=inner_lr, weight_decay=inner_wd)
                if fair != 'none':
                    client_optimizers_lambda[i] = torch.optim.Adam(constraints[i].parameters(), lr=inner_lr, weight_decay=inner_wd)

            hnet.to(device)

            optimizer = torch.optim.Adam(params=hnet.parameters(), lr=lr, weight_decay=wd)
            loss = torch.nn.BCELoss()
            step_iter = trange(steps)

            for round in step_iter:
                if round == 0:
                    step_results, avg_acc_all, all_acc, eod, spd = eval_model(nodes=training_nodes_test, num_nodes=num_nodes-10, hnet=hnet,
                                                                              model=models, device=device,
                                                                              which_position=which_position)

                    c_acc_p_epoch.append(all_acc)
                    acc_p_epoch.append(avg_acc_all)

                    spd_clients = []
                    eod_clients = []

                    for i in range(len(spd)):
                        if i % 2 == 0:
                            spd_clients.append(spd[i])
                        elif i % 2 == 1:
                            eod_clients.append(eod[i])

                    c_spd_p_epoch.append(spd_clients)
                    spd_p_epoch.append(np.mean(spd_clients))

                    c_eod_p_epoch.append(eod_clients)
                    eod_p_epoch.append(np.mean(eod_clients))
                hnet.train()
                node_id = random.choice(range(num_nodes-10))

                model = models[node_id]

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

                    batch = next(iter(training_nodes_train[node_id]))
                    x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
                    s = x[:, which_position].to(device)

                    pred, m_mu_q = model(x, s, y)
                    if fair == 'none':
                        err = loss(pred, y.unsqueeze(1))
                    else:
                        l = loss(pred, y.unsqueeze(1))
                        c = constraint(m_mu_q)
                        er = l + c
                        err = er.mean()

                    err.backward()
                    inner_optim_theta.step()

                    if fair != 'none':
                        constraint.lmbda.data = torch.clamp(constraint.lmbda.data, min=0)
                        torch.nn.utils.clip_grad_norm_(constraint.lmbda.data, b, norm_type=1)

                        if torch.nn.utils.clip_grad_norm_(constraint.lmbda.data, b, norm_type=1) > b:
                            print(torch.nn.utils.clip_grad_norm_(constraint.lmbda.data, b, norm_type=1))
                            print(constraint.lmbda)
                            exit(1)

                        for i, item in enumerate(constraint.lmbda.data):
                            if item < 0:
                                print(constraint.lmbda)
                                exit(2)

                        for group in inner_optim_lambda.param_groups:
                            for p in group['params']:
                                p.grad = -1 * p.grad

                        inner_optim_lambda.step()

                optimizer.zero_grad()
                final_state = model.state_dict()
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
                hnet_grads = torch.autograd.grad(list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()))
                for p, g in zip(hnet.parameters(), hnet_grads):
                    p.grad = g

                optimizer.step()

            step_results, avg_acc_all, all_acc, eod, spd = eval_model(nodes=training_nodes_test, num_nodes=num_nodes-10, hnet=hnet, model=models, device=device, which_position=which_position)

            c_acc_p_epoch.append(all_acc)
            acc_p_epoch.append(avg_acc_all)

            spd_clients = []
            eod_clients = []

            for i in range(len(spd)):
                if i % 2 == 0:
                    spd_clients.append(spd[i])
                elif i % 2 == 1:
                    eod_clients.append(eod[i])

            c_spd_p_epoch.append(spd_clients)
            spd_p_epoch.append(np.mean(spd_clients))

            c_eod_p_epoch.append(eod_clients)
            eod_p_epoch.append(np.mean(eod_clients))

            # For multi-round
            all_acc_10.append(avg_acc_all)
            all_eod_10.append(np.mean(eod_clients))
            all_spd_10.append(np.mean(spd_clients))

            final_acc = np.mean(all_acc_10)
            final_eod = np.mean(all_eod_10)
            final_spd = np.mean(all_spd_10)

            print(
                f"Final Results Training Clients| AVG Acc: {avg_acc_all:.4f} | AVG EOD: {np.mean(eod_clients):.4f} | AVG SPD: {np.mean(spd_clients):.4f} | TV: {nodes.total_variation[0]}")


            # Train and test 10 remaining clients for 500 rounds
            print('Starting Novel Client Training')

            all_acc_10 = []
            all_eod_10 = []
            all_spd_10 = []

            c_acc_p_epoch = []
            acc_p_epoch = []

            c_eod_p_epoch = []
            eod_p_epoch = []

            c_spd_p_epoch = []
            spd_p_epoch = []

            training_state_dict = hnet.state_dict()

            hnet = LRHyper(device=device, n_nodes=num_nodes, embedding_dim=embed_dim,
                           context_vector_size=num_features,
                           hidden_size=num_features, hnet_hidden_dim=hyper_hid, hnet_n_hidden=n_hidden)

            new_client_addition = torch.rand((10,13)).to(device)

            with torch.no_grad():
                extended_embedding = training_state_dict['embeddings.weight']
                extended_embedding = torch.cat((extended_embedding, new_client_addition), dim=0)
                hnet.embeddings.weight.copy_(extended_embedding)

                hnet.mlp[0].weight.copy_(training_state_dict['mlp.0.weight'])
                hnet.mlp[0].bias.copy_(training_state_dict['mlp.0.bias'])

                hnet.mlp[2].weight.copy_(training_state_dict['mlp.2.weight'])
                hnet.mlp[2].bias.copy_(training_state_dict['mlp.2.bias'])

                hnet.mlp[4].weight.copy_(training_state_dict['mlp.4.weight'])
                hnet.mlp[4].bias.copy_(training_state_dict['mlp.4.bias'])

                hnet.mlp[6].weight.copy_(training_state_dict['mlp.6.weight'])
                hnet.mlp[6].bias.copy_(training_state_dict['mlp.6.bias'])

                hnet.mlp[8].weight.copy_(training_state_dict['mlp.8.weight'])
                hnet.mlp[8].bias.copy_(training_state_dict['mlp.8.bias'])

                hnet.fc1_weights.weight.copy_(training_state_dict['fc1_weights.weight'])
                hnet.fc1_weights.bias.copy_(training_state_dict['fc1_weights.bias'])
                hnet.fc1_bias.weight.copy_(training_state_dict['fc1_bias.weight'])
                hnet.fc1_bias.bias.copy_(training_state_dict['fc1_bias.bias'])

            for name, param in hnet.named_parameters():
                if name != 'embeddings.weight':
                    param.requires_grad = False
                else:
                    update_layer = param

            hnet.to(device)
            step_iter = trange(500)
            for round in step_iter:
                if round == 0:
                    step_results, avg_acc_all, all_acc, eod, spd = eval_model(nodes=novel_nodes_test, num_nodes=10, hnet=hnet,
                                                                              model=models, device=device,
                                                                              which_position=which_position)

                    c_acc_p_epoch.append(all_acc)
                    acc_p_epoch.append(avg_acc_all)

                    spd_clients = []
                    eod_clients = []

                    for i in range(len(spd)):
                        if i % 2 == 0:
                            spd_clients.append(spd[i])
                        elif i % 2 == 1:
                            eod_clients.append(eod[i])

                    c_spd_p_epoch.append(spd_clients)
                    spd_p_epoch.append(np.mean(spd_clients))

                    c_eod_p_epoch.append(eod_clients)
                    eod_p_epoch.append(np.mean(eod_clients))
                hnet.train()
                client_values = [90,91,92,93,94,95,96,97,98,99]
                node_id = np.random.choice(client_values)
                place_id = client_values.index(node_id)
                model = models[node_id]

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

                for j in range(10):
                    inner_optim_theta.zero_grad()
                    if fair != 'none':
                        inner_optim_lambda.zero_grad()
                    optimizer.zero_grad()

                    batch = next(iter(novel_nodes_train[place_id]))
                    x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
                    s = x[:, which_position].to(device)

                    pred, m_mu_q = model(x, s, y)
                    if fair == 'none':
                        err = loss(pred, y.unsqueeze(1))
                    else:
                        l = loss(pred, y.unsqueeze(1))
                        c = constraint(m_mu_q)
                        er = l + c
                        err = er.mean()

                    err.backward()
                    inner_optim_theta.step()

                    if fair != 'none':
                        constraint.lmbda.data = torch.clamp(constraint.lmbda.data, min=0)
                        torch.nn.utils.clip_grad_norm_(constraint.lmbda.data, b, norm_type=1)

                        if torch.nn.utils.clip_grad_norm_(constraint.lmbda.data, b, norm_type=1) > b:
                            print(torch.nn.utils.clip_grad_norm_(constraint.lmbda.data, b, norm_type=1))
                            print(constraint.lmbda)
                            exit(1)

                        for i, item in enumerate(constraint.lmbda.data):
                            if item < 0:
                                print(constraint.lmbda)
                                exit(2)

                        for group in inner_optim_lambda.param_groups:
                            for p in group['params']:
                                p.grad = -1 * p.grad

                        inner_optim_lambda.step()

                optimizer.zero_grad()
                final_state = model.state_dict()
                delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

                hnet_grads = torch.autograd.grad(list(weights.values()), update_layer, grad_outputs=list(delta_theta.values()))
                for p, g in zip(hnet.parameters(), hnet_grads):
                    p.grad = g

                optimizer.step()
            step_results, avg_acc_all, all_acc, eod, spd = eval_model(nodes=novel_nodes_test, num_nodes=10,
                                                                      hnet=hnet, model=models, device=device,
                                                                      which_position=which_position)

            c_acc_p_epoch.append(all_acc)
            acc_p_epoch.append(avg_acc_all)

            spd_clients = []
            eod_clients = []

            for i in range(len(spd)):
                if i % 2 == 0:
                    spd_clients.append(spd[i])
                elif i % 2 == 1:
                    eod_clients.append(eod[i])

            c_spd_p_epoch.append(spd_clients)
            spd_p_epoch.append(np.mean(spd_clients))

            c_eod_p_epoch.append(eod_clients)
            eod_p_epoch.append(np.mean(eod_clients))

            # For multi-round
            all_acc_10.append(avg_acc_all)
            all_eod_10.append(np.mean(eod_clients))
            all_spd_10.append(np.mean(spd_clients))

            final_acc_novel = np.mean(all_acc_10)
            final_eod_novel = np.mean(all_eod_10)
            final_spd_novel = np.mean(all_spd_10)

            total_variation = nodes.total_variation

            avg_acc_difference = final_acc_novel - final_acc
            avg_eod_difference = final_eod_novel - final_eod
            avg_spd_difference = final_spd_novel - final_spd

            print(f"Final Results Novel Clients| AVG Acc: {avg_acc_all:.4f} | AVG EOD: {np.mean(eod_clients):.4f} | AVG SPD: {np.mean(spd_clients):.4f}")

            TV.append(total_variation[0])
            avg_acc_difference_all.append(avg_acc_difference)
            avg_eod_difference_all.append(avg_eod_difference)
            avg_spd_difference_all.append(avg_spd_difference)

        repeated_acc.append(np.mean(avg_acc_difference_all))
        repeated_eod.append(np.mean(avg_eod_difference_all))
        repeated_spd.append(np.mean(avg_spd_difference_all))
        tv_f.append(np.mean(TV))

    sns.set(font_scale=1.5)

    plt.plot(tv_f, repeated_acc, 'b-', linewidth=2, label='Acc')
    plt.plot(tv_f, repeated_eod, 'r-', linewidth=2, label='EOD')
    plt.plot(tv_f, repeated_spd, 'g', linewidth=2, label='SPD')
    plt.xlabel('Total Variation (nats)')
    plt.ylabel('Generalization Gap')
    #plt.ylim([-1, 1])
    plt.tight_layout()
    plt.show()


def main():
    file = open("/home/ancarey/FairFLHN/large-federation/10-run-results.txt", "w")
    file.close()

    num_clients = [100]#[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for i, c in enumerate(num_clients):
        pd.set_option('display.float_format', lambda x: '%.1f' % x)

        parser = argparse.ArgumentParser(description="Fair Hypernetworks")

        parser.add_argument("--data_name", type=str, default="adult", choices=["adult", "compas"], help="choice of dataset")
        parser.add_argument("--num_nodes", type=int, default=c, help="number of simulated clients")
        parser.add_argument("--num_steps", type=int, default=5000)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--inner_steps", type=int, default=10, help="number of inner steps")
        parser.add_argument("--n_hidden", type=int, default=4, help="num. hidden layers")
        parser.add_argument("--inner_lr", type=float, default=.001, help="learning rate for inner optimizer")
        parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
        parser.add_argument("--wd", type=float, default=1e-10, help="weight decay")
        parser.add_argument("--inner_wd", type=float, default=1e-10, help="inner weight decay")
        parser.add_argument("--hyper_hid", type=int, default=100, help="hypernet hidden dim")
        parser.add_argument("--seed", type=int, default=0, help="seed value")
        parser.add_argument("--fair", type=str, default="both", choices=["none", "eo", "dp", "both"],
                            help="whether to use fairness of not.")
        parser.add_argument("--alpha", type=int, default=[.01,.01], help="fairness/accuracy trade-off parameter")
        parser.add_argument("--which_position", type=int, default=8, choices=[5, 8],
                            help="which position the sensitive attribute is in. 5: compas, 8: adult")
        args = parser.parse_args()
        set_logger()
        device = "cuda:2"
        print("\n====================")
        print("| Starting Training |")
        print("=====================")
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
            alpha_all=args.alpha,
            fair=args.fair,
            which_position=args.which_position)


if __name__ == "__main__":
    main()