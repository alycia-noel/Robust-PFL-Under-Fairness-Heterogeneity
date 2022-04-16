import argparse
from pathlib import Path
import logging
import json
import random
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import pandas as pd
import torch.utils.data
from tqdm import trange
from experiments.adult.models import NN_Context, NNHyper
from experiments.adult.node import BaseNodes
from experiments.adult.utils import get_device, seed_everything, set_logger
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

def eval_model(nodes, num_nodes, hnet, model, loss, device, split, confusion=False):
    curr_results, pred, true = evaluate(nodes, num_nodes, hnet, model, loss, device, split=split)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    if confusion:
        classes = {'No', 'Yes'}
        for i in range(len(pred)):
            cf_matrix = confusion_matrix(true[i], pred[i])
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                                 columns=[i for i in classes])
            plt.figure(figsize=(12, 7))
            sn.heatmap(df_cm, annot=True)
            title = 'Confusion Matrix for Client ' + str(i + 1)
            plt.title(title)
            plt.show()

    return curr_results, avg_loss, avg_acc, all_acc

@torch.no_grad()
def evaluate(nodes, num_nodes, hnet, model, loss, device, split='test'):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))
    preds = []
    true = []
    for node_id in range(num_nodes):
        pred_client = []
        true_client = []
        running_loss, running_correct, running_samples = 0, 0, 0
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
            print(x, y)

            true_client.extend(y.cpu().numpy())

            avg_context_vector = model(x.to(device), context_only=True)
            weights = hnet(avg_context_vector, torch.tensor([node_id], dtype=torch.long).to(device))
            net_dict = model.state_dict()
            hnet_dict = {k: v for k, v in weights.items() if k in net_dict}
            net_dict.update(hnet_dict)
            model.load_state_dict(net_dict)

            pred = model(x, context_only = False)
            pred_prob = torch.sigmoid(pred)
            pred_thresh = (pred_prob > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())
            running_loss += loss(pred, y.unsqueeze(1)).item()
            correct = torch.eq(pred_thresh,y.unsqueeze(1)).type(torch.cuda.LongTensor)
            running_correct += torch.count_nonzero(correct).item()

            running_samples += len(y)

        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples
        preds.append(pred_client)
        true.append(true_client)
        exit(1)
    return results, preds, true

def train(data_name, model_name, classes_per_node, num_nodes, steps, inner_steps, lr, inner_lr, wd, inner_wd, hyper_hid, n_hidden, bs, device, eval_every, seed, fair, save_path):

    if fair == "none":

        nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node)

        num_features = len(nodes.features)

        embed_dim = num_features

        if model_name == 'NN':
            hnet = NNHyper(n_nodes=num_nodes, embedding_dim=embed_dim, context_vector_size=num_features, hidden_size=num_features, hnet_hidden_dim = hyper_hid, hnet_n_hidden=n_hidden)
            model = NN_Context(input_size=num_features, context_vector_size=num_features, context_hidden_size=50, nn_hidden_size=num_features, dropout=.5)

        hnet.to(device)
        model.to(device)

        optimizer = torch.optim.Adam(params=hnet.parameters(), lr=lr, weight_decay=wd)
        loss = torch.nn.BCEWithLogitsLoss()

        last_eval = -1
        best_step = -1
        best_acc = -1
        test_best_based_on_step, test_best_min_based_on_step = -1, -1
        test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
        step_iter = trange(steps)

        results = defaultdict(list)
        i = 0
        for step in step_iter:
            hnet.train()
            node_id = random.choice(range(num_nodes))

            for j in range(inner_steps):
                batch = next(iter(nodes.train_loaders[node_id]))
                x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
                print(torch.max(x, dim=0), torch.min(x,  dim=0))
                if j == 0:
                    avg_context_vector = model(x, context_only=True)
                    weights = hnet(avg_context_vector, torch.tensor([node_id], dtype=torch.long).to(device))
                    net_dict = model.state_dict()
                    hnet_dict = {k: v for k, v in weights.items() if k in net_dict}
                    net_dict.update(hnet_dict)
                    model.load_state_dict(net_dict)

                    inner_optim = torch.optim.Adam(model.parameters(), lr=inner_lr, weight_decay=inner_wd)

                    inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

                    with torch.no_grad():
                        model.eval()
                        pred = model(x, context_only = False)
                        prvs_loss = loss(pred, y.unsqueeze(1))
                        pred_prob = torch.sigmoid(pred)
                        pred_thresh = (pred_prob > 0.5).long()
                        correct = torch.eq(pred_thresh, y.unsqueeze(1)).type(torch.cuda.LongTensor)
                        prvs_acc = torch.count_nonzero(correct).item() / len(pred)
                        model.train()

                model.train()
                inner_optim.zero_grad()
                optimizer.zero_grad()

                pred = model(x, context_only=False)
                err = loss(pred, y.unsqueeze(1))
                err.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
                inner_optim.step()

                i += 1

            optimizer.zero_grad()
            final_state = model.state_dict()
            delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
            hnet_grads = torch.autograd.grad(list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()))

            for p, g in zip(hnet.parameters(), hnet_grads):
                p.grad = g

            torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
            optimizer.step()

            step_iter.set_description(f"Step: {step + 1}, Node ID: {node_id}, Loss: {prvs_loss:.4f},  Acc: {prvs_acc:.4f}")

            if step % eval_every == 0:
                last_eval = step
                step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, model, loss, device, confusion=True,
                                                                      split="test")

                logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
                writer.add_scalar('testing accuracy', avg_acc, step)
                results['test_avg_loss'].append(avg_loss)
                results['test_avg_acc'].append(avg_acc)

                # _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, model, loss, device, split="val")
                # if best_acc < val_avg_acc:
                #     best_acc = val_avg_acc
                #     best_step = step
                #     test_best_based_on_step = avg_acc
                #     test_best_min_based_on_step = np.min(all_acc)
                #     test_best_max_based_on_step = np.max(all_acc)
                #     test_best_std_based_on_step = np.std(all_acc)

                # results['val_avg_loss'].append(val_avg_loss)
                # results['val_avg_acc'].append(val_avg_acc)
                results['best_step'].append(best_step)
                results['best_val_acc'].append(best_acc)
                results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
                results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
                results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
                results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

if __name__ == '__main__':
    writer = SummaryWriter('results')

    parser = argparse.ArgumentParser(description="Fair Hypernetworks")

    parser.add_argument("--data_name", type=str, default="compas", choices=["adult", "compas"], help="choice of dataset")
    parser.add_argument("--model_name", type=str, default="NN", choices=["NN", "LR"], help="choice of model")
    parser.add_argument("--num_nodes", type=int, default=4, help="number of simulated clients")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--inner_steps", type=int, default=50, help="number of inner steps")
    parser.add_argument("--n_hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner_lr", type=float, default=1e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-8, help="weight decay")
    parser.add_argument("--inner_wd", type=float, default=1e-8, help="inner weight decay")
    parser.add_argument("--embed_dim", type=int, default=10, help="embedding dim")
    parser.add_argument("--hyper_hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--gpu", type=int, default=4, help="gpu device ID")
    parser.add_argument("--eval_every", type=int, default=50, help="eval every X selected epochs")
    parser.add_argument("--save_path", type=str, default="/home/ancarey/FairFLHN/experiments/adult/results", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=0, help="seed value")
    parser.add_argument("--fair", type=str, default="none", choices=["none", "eo", "dp"], help="whether to use fairness of not.")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count()
    set_logger()
    seed_everything(args.seed)

    device = get_device(gpus=args.gpu)

    args.classes_per_node = 1

    train(data_name=args.data_name,
          model_name=args.model_name,
          classes_per_node = args.classes_per_node,
          num_nodes=args.num_nodes,
          steps=args.num_steps,
          inner_steps=args.inner_steps,
          lr = args.lr,
          inner_lr = args.inner_lr,
          wd = args.wd,
          inner_wd = args.inner_wd,
          hyper_hid = args.hyper_hid,
          n_hidden = args.n_hidden,
          bs = args.batch_size,
          device = device,
          eval_every = args.eval_every,
          seed = args.seed,
          fair = args.fair,
          save_path = args.save_path)
