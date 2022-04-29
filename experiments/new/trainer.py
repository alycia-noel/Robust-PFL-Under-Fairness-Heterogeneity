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
from experiments.new.models import LR, Context, LRHyper
from experiments.new.node import BaseNodes
from experiments.new.utils import get_device, seed_everything, set_logger, TP_FP_TN_FN, metrics
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sn
from fairtorch import DemographicParityLoss, EqualiedOddsLoss
warnings.filterwarnings("ignore")


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

def eval_model(nodes, num_nodes, hnet, model, cnet, num_features, loss, device, fair, fair_loss,confusion, which_position):
    curr_results, pred, true, f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd = evaluate(nodes, num_nodes, hnet, model, cnet, num_features, loss, device, fair, fair_loss, which_position)
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
def evaluate(nodes, num_nodes, hnet, model, cnet, num_features, loss, device, fair, fair_loss, which_position):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))
    preds = []
    true = []
    f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd = [], [], [], [], [], [], [], [], []

    for node_id in range(num_nodes):
        pred_client = []
        true_client = []
        queries_client = []

        running_loss, running_correct, running_samples = 0, 0, 0

        curr_data = nodes.test_loaders[node_id]


        for batch_count, batch in enumerate(curr_data):
            x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
            s = x[:, which_position].to(device)
            true_client.extend(y.cpu().numpy())
            queries_client.extend(x.cpu().numpy())

            avg_context_vector = cnet(x)
            weights = hnet(avg_context_vector, torch.tensor([node_id], dtype=torch.long).to(device))
            model.load_state_dict(weights)

            prediction_vector = avg_context_vector.expand(len(x), num_features)
            prediction_vector = torch.cat((prediction_vector, x), dim=1)
            pred = model(prediction_vector)

            pred_prob = torch.sigmoid(pred)
            pred_thresh = (pred_prob > 0.5).long()
            pred_client.extend(pred_thresh.flatten().cpu().numpy())

            if fair == 'none':
                running_loss += loss(pred, y.unsqueeze(1)).item()
            else:
                running_loss += (loss(pred, y.unsqueeze(1)) + fair_loss(x, pred, s, y).to(device)).item()

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
        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples
        preds.append(pred_client)
        true.append(true_client)

    return results, preds, true, f1, f1_f, f1_m, a, f_a, m_a, aod, eod, spd

def train(writer, device, data_name,model_name,classes_per_node,num_nodes,steps,inner_steps,lr,inner_lr,wd,inner_wd, hyper_hid,n_hidden,bs, alpha,fair, which_position):

    seed_everything(0)

    nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node)

    num_features = len(nodes.features)

    embed_dim = num_features

    hnet = LRHyper(device=device,n_nodes=num_nodes, embedding_dim=embed_dim, context_vector_size=num_features, hidden_size=num_features, hnet_hidden_dim=hyper_hid, hnet_n_hidden=n_hidden)
    model = LR(input_size=num_features)
    cnet = Context(input_size=num_features, context_vector_size=num_features, context_hidden_size=50)

    combo_params = list(model.parameters()) + list(cnet.parameters())

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:4"

    hnet.to(device)
    model.to(device)
    cnet.to(device)

    optimizer = torch.optim.Adam(params=hnet.parameters(), lr=lr, weight_decay=wd)
    loss = torch.nn.BCEWithLogitsLoss()

    if fair == 'none':
        fair_loss = None
    elif fair == 'dp':
        fair_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=alpha)
    elif fair == 'eo':
        fair_loss = EqualiedOddsLoss(sensitive_classes=[0, 1], alpha=alpha)
    else:
        fair_loss = None

    step_iter = trange(steps)

    for step in step_iter:
        hnet.train()
        node_id = random.choice(range(num_nodes))

        node_c_i = nodes.c_i[node_id]

        weights = hnet(node_c_i, torch.tensor([node_id], dtype=torch.long).to(device))
        model.load_state_dict(weights)

        inner_optim = torch.optim.Adam(combo_params, lr=inner_lr, weight_decay=inner_wd)

        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

        if node_id % 2 == 0:
            fair_loss = DemographicParityLoss(sensitive_classes=[0, 1], alpha=100)
        else:
            fair_loss = EqualiedOddsLoss(sensitive_classes=[0, 1], alpha=150)

        avg_c_i = []
        running_fair_loss = []

        for j in range(inner_steps):
            batch = next(iter(nodes.train_loaders[node_id]))
            x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
            s = x[:,which_position].to(device)

            model.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()

            avg_context_vector = cnet(x)
            prediction_vector = avg_context_vector.expand(len(x), num_features)
            prediction_vector = torch.cat((prediction_vector, x), dim=1)

            pred = model(prediction_vector)
            avg_c_i.append(avg_context_vector)

            if fair == 'none':
                err = loss(pred, y.unsqueeze(1))
            else:
                fair_l = fair_loss(x, pred, s, y)
                running_fair_loss.append(fair_l.item())
                err = loss(pred, y.unsqueeze(1)) + fair_l.to(device)

            err.backward()

            # plot_grad_flow(model.named_parameters())
            # plot_grad_flow(cnet.named_parameters())

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            inner_optim.step()

        nodes.c_i[node_id] = torch.cuda.FloatTensor([sum(sub_list) / len(sub_list) for sub_list in zip(*avg_c_i)])

        optimizer.zero_grad()
        final_state = model.state_dict()
        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
        hnet_grads = torch.autograd.grad(list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values()))

        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g

        #torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        if step % 49 == 0 or step == 1999 or step == 0:
            step_results, avg_loss, avg_acc, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd = eval_model(nodes, num_nodes, hnet, model, cnet, num_features, loss, device, confusion=False, fair=fair, fair_loss=fair_loss, which_position=which_position)

            logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}, Fairness: {fair_loss}")
            writer.add_scalars('testing accuracy', {
                'average': avg_acc,
                'client 1': all_acc[0],
                'client 2': all_acc[1],
                'client 3': all_acc[2],
                'client 4': all_acc[3]
            }, step)
            writer.add_scalars('testing loss', {
                'average': avg_loss,
                'client 1': all_loss[0],
                'client 2': all_loss[1],
                'client 3': all_loss[2],
                'client 4': all_loss[3]
            }, step)
            writer.add_scalars('AOD', {
                'average': np.mean(aod),
                'client 1': aod[0],
                'client 2': aod[1],
                'client 3': aod[2],
                'client 4': aod[3]
            }, step)
            writer.add_scalars('EOD', {
                'average': np.mean(eod),
                'client 1': eod[0],
                'client 2': eod[1],
                'client 3': eod[2],
                'client 4': eod[3]
            }, step)
            writer.add_scalars('SPD', {
                'average': np.mean(spd),
                'client 1': spd[0],
                'client 2': spd[1],
                'client 3': spd[2],
                'client 4': spd[3]
            }, step)


    step_results, avg_loss, avg_acc, all_acc, all_loss, f1, f1_f, f1_m, f_a, m_a, aod, eod, spd = eval_model(nodes, num_nodes, hnet, model, cnet, num_features, loss, device, confusion=False,fair=fair, fair_loss = fair_loss, which_position=which_position)
    logging.info(f"\n\nFinal Results | AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
    for i in range(num_nodes):
        print("\nClient", i+1)
        print(i)
        print(f"Acc: {all_acc[i]:.4f}, F Acc: {f_a[i]:.4f}, M Acc: {m_a[i]:.4f}, F1: {f1[i]:.4f}, AOD: {aod[i]:.4f}, EOD: {eod[i]:.4f}, SPD: {spd[i]:.4f}")


def main():
    pd.set_option('display.float_format', lambda x: '%.1f' % x)

    writer = SummaryWriter('results')

    parser = argparse.ArgumentParser(description="Fair Hypernetworks")

    parser.add_argument("--data_name", type=str, default="compas", choices=["adult", "compas"], help="choice of dataset")
    parser.add_argument("--model_name", type=str, default="LR", choices=["NN", "LR"], help="choice of model")
    parser.add_argument("--num_nodes", type=int, default=4, help="number of simulated clients")
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--inner_steps", type=int, default=50, help="number of inner steps")
    parser.add_argument("--n_hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner_lr", type=float, default=1e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--inner_wd", type=float, default=1e-7, help="inner weight decay")
    parser.add_argument("--embed_dim", type=int, default=10, help="embedding dim")
    parser.add_argument("--hyper_hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--gpu", type=int, default=4, help="gpu device ID")
    parser.add_argument("--eval_every", type=int, default=50, help="eval every X selected epochs")
    parser.add_argument("--save_path", type=str, default="/home/ancarey/FairFLHN/experiments/adult/results", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=0, help="seed value")
    parser.add_argument("--fair", type=str, default="both", choices=["none", "eo", "dp", "both"], help="whether to use fairness of not.")
    parser.add_argument("--alpha", type=int, default=150, help="fairness/accuracy trade-off parameter")
    parser.add_argument("--which_position", type=int, default=5, choices=[5,8], help="which position the sensitive attribute is in. 5: compas, 8: adult")
    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count()
    set_logger()

    device = get_device(gpus=args.gpu)

    args.classes_per_node = 2

    train(
    writer,
    device=device,
    data_name=args.data_name,
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
    alpha = args.alpha,
    fair = args.fair,
    which_position = args.which_position)

if __name__ == "__main__":
    main()