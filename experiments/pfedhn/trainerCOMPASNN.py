import argparse
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
import torch.utils.data
from tqdm import trange # for the progress bar

from experiments import HyperCOMPASNN, TargetAndContextCOMPASNN # loads the model architecture
from experiments import BaseNodes # loads the client generator
from experiments.dp_one_client_code.utils import get_device, set_logger, set_seed  # loads extra tools

import warnings
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

# The function to return the evaluation results of the model to the client.
# nodes: total number of clients
# hnet: the hypernetwork model
# net: the local model
# criteria: the loss function
# device: GPU or CPU
# split: test (but can also be train)
# return: current_results, average loss, average accuracy, and overall accuracy so far
def eval_model(nodes, num_nodes, hnet, combonet, criteria, device, split):
    curr_results = evaluate(nodes, num_nodes, hnet, combonet, criteria, device, split=split)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    return curr_results, avg_loss, avg_acc, all_acc

# The function to actually evaluate the model (i.e., test because there is no backpropagation)
@torch.no_grad()
def evaluate(nodes: BaseNodes, num_nodes, hnet, combonet, criteria, device, split='test'):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))

    for node_id in range(num_nodes):  # iterating over nodes

        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        elif split == 'val':
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.float().to(device) for t in batch)
            _, avg_context_vector, prediction_vector = combonet(img, contextonly=True)
            weights_1 = hnet(avg_context_vector)
            net_dict = combonet.state_dict()
            hnet_dict = {k: v for k, v in weights_1.items() if k in net_dict}
            net_dict.update(hnet_dict)
            combonet.load_state_dict(net_dict)
            pred = combonet(img, contextonly=False)
            running_loss = running_loss +  criteria(pred, label.unsqueeze(1)).mean().item()
            running_correct = running_correct + pred.argmax(1).eq(label).sum().item()
            running_samples = running_samples + len(label)

        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples

    return results

# Training function - for both the client and hypernet
def train(data_name: str, num_nodes: int, steps: int, inner_steps: int, device, eval_every: int, save_path: Path, seed: int) -> None:

    ###############################
    # init nodes, hnet, local net #
    ###############################
    # generate the clients
    nodes = BaseNodes(data_name, data_path=None, n_nodes = 2, classes_per_node=1, batch_size=16)

    embed_dim=13

    # create the hypernet, local, and context network
    hnet = HyperCOMPASNN(embed_dim, hidden_dim=64)
    combonet = TargetAndContextCOMPASNN(no_features = 8, hidden_sizes=[64,64,32], dropout_rate = .2, vector_size = 13)

    # send both of the networks to the GPU
    hnet = hnet.to(device)
    combonet = combonet.to(device)

    # setting up optimizers
    optimizer = torch.optim.Adam(hnet.parameters(), lr=2.e-4, weight_decay=0.)

    # Loss function
    criteria = nn.BCELoss(reduction='mean')  # Binary Cross Entropy loss

    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)

    results = defaultdict(list)
    for step in step_iter:
        hnet.train()

        # select client at random
        node_id = random.choice(range(num_nodes))

        batch = next(iter(nodes.test_loaders[node_id]))

        features, label = tuple(t.float().to(device) for t in batch)

        context_vectors, avg_context_vector, prediction_vector = combonet(features, contextonly=True)

        # produce & load local network weights
        weights = hnet(avg_context_vector)
        net_dict = combonet.state_dict()
        hnet_dict = {k: v for k, v in weights.items() if k in net_dict}
        net_dict.update(hnet_dict)
        combonet.load_state_dict(net_dict)

        # init inner optimizer
        inner_optim = torch.optim.Adam(combonet.parameters(), lr=5e-3, weight_decay=0)

        # storing theta_i for later calculating delta theta
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

        # NOTE: evaluation on sent model
        with torch.no_grad():
            combonet.eval()
            pred = combonet(features, contextonly=False)
            prvs_loss = criteria(pred, label.unsqueeze(1))
            prvs_acc = pred.argmax(1).eq(label).sum().item() / len(label)
            combonet.train()

        # inner updates -> obtaining theta_tilda
        for i in range(inner_steps):
            combonet.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()

            # Create context vector for this batch
            batch = next(iter(nodes.train_loaders[node_id]))
            features, label = tuple(t.float().to(device) for t in batch)

            pred = combonet(features, contextonly=False)

            loss = criteria(pred, label.unsqueeze(1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(combonet.parameters(), 50)

            inner_optim.step()

        optimizer.zero_grad()

        final_state = combonet.state_dict()

        # calculating delta theta
        delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

        #######################
        # Hypernetwork Update #
        #######################

        # calculating phi gradient
        hnet_grads = torch.autograd.grad(
            list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
        )

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        step_iter.set_description(
            f"Step: {step+1}, Node ID: {node_id}, Loss: {prvs_loss:.4f},  Acc: {prvs_acc:.4f}"
        )

        if step % eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, combonet, criteria, device, split="test")
            logging.info(f"\nStep: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, combonet, criteria, device, split="val")
            if best_acc < val_avg_acc:
                best_acc = val_avg_acc
                best_step = step
                test_best_based_on_step = avg_acc
                test_best_min_based_on_step = np.min(all_acc)
                test_best_max_based_on_step = np.max(all_acc)
                test_best_std_based_on_step = np.std(all_acc)

            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_acc)
            results['best_step'].append(best_step)
            results['best_val_acc'].append(best_acc)
            results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
            results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
            results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
            results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

    if step != last_eval:
        _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, combonet, criteria, device, split="val")
        step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, combonet, criteria, device, split="test")
        logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

        results['test_avg_loss'].append(avg_loss)
        results['test_avg_acc'].append(avg_acc)

        if best_acc < val_avg_acc:
            best_acc = val_avg_acc
            best_step = step
            test_best_based_on_step = avg_acc
            test_best_min_based_on_step = np.min(all_acc)
            test_best_max_based_on_step = np.max(all_acc)
            test_best_std_based_on_step = np.std(all_acc)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['best_step'].append(best_step)
        results['best_val_acc'].append(best_acc)
        results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
        results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
        results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
        results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / f"results_{inner_steps}_inner_steps_seed_{seed}.json"), "w") as file:
        json.dump(results, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Hypernetwork")

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument("--data-name", type=str, default="COMPAS", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=2, help="number of simulated nodes")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--inner-steps", type=int, default=50, help="number of inner steps")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="/home/ancarey/adaptive-hypernets-results/COMPAS", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    train(
        data_name=args.data_name,
        num_nodes=args.num_nodes,
        steps=args.num_steps,
        inner_steps=args.inner_steps,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed
    )
