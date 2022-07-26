import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3, 4, 5, 6, 7"
import warnings
import torch
import torch.utils.data
from tqdm import trange
from experiments.utils import seed_everything, TP_FP_TN_FN, metrics
from experiments.models import LR, Constraint
from experiments.node import BaseNodes
warnings.filterwarnings("ignore")

@torch.no_grad()
def evaluate(model, device, which_position, test_loader):
    preds = []
    true = []
    queries = []

    model.eval()
    model.to(device)

    running_loss, running_correct, running_samples = 0, 0, 0

    for batch_count, batch in enumerate(test_loader):
        x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
        s = x[:, which_position].to(device)

        true.extend(y.cpu().numpy())
        queries.extend(x.cpu().numpy())

        pred, m_mu_q = model(x, s, y)
        pred_thresh = (pred > 0.5).long()
        preds.extend(pred_thresh.flatten().cpu().numpy())

        correct = torch.eq(pred_thresh, y.unsqueeze(1)).type(torch.cuda.LongTensor)
        running_correct += torch.count_nonzero(correct).item()

        running_samples += len(y)

    tp, fp, tn, fn = TP_FP_TN_FN(queries, preds, true, which_position)
    print(tp, fp, tn, fn)
    accuracy, f_acc, m_acc, eod, spd = metrics(tp, fp, tn, fn)

    return accuracy, eod, spd


def train(device, steps, lr, wd, alpha, fair, which_position, num_features, train_loader, test_loader, client_num):
    seed_everything(0)
    b = 1/alpha

    model = LR(input_size=num_features, bound=alpha, fairness=fair)
    constraint= Constraint(fair=fair, bound=alpha)
    client_optimizers_theta = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if fair != 'none':
        client_optimizers_lambda = torch.optim.Adam(constraint.parameters(), lr=lr, weight_decay=wd)

    loss = torch.nn.BCELoss()

    model.train()
    model.to(device)
    if fair != 'none':
        constraint.to(device)

    step_iter = trange(steps)
    for j in step_iter:
        client_optimizers_theta.zero_grad()

        if fair != 'none':
            client_optimizers_lambda.zero_grad()

        batch = next(iter(train_loader))
        x, y = tuple((t.type(torch.cuda.FloatTensor)).to(device) for t in batch)
        s = x[:, which_position].to(device)

        pred, m_mu_q = model(x, s, y)

        if fair == 'none':
            err = loss(pred, y.unsqueeze(1))
        else:
            err = (loss(pred, y.unsqueeze(1)) + constraint(m_mu_q)).mean()

        err.backward()
        client_optimizers_theta.step()

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

            for group in client_optimizers_lambda.param_groups:
                for p in group['params']:
                    p.grad = -1 * p.grad

            client_optimizers_lambda.step()

    accuracy, eod, spd = evaluate(model, device, which_position, test_loader)
    print(f"Acc: {accuracy:.4f}, EOD: {eod:.4f}, SPD: {spd:.4f}")


def main():
    seed_everything(0)

    alpha = 1
    num_nodes = 4
    fairness = 'both'
    data_name = 'adult'
    classes_per_node = 2
    device = 'cuda:5'
    num_steps = 5000
    wd = 1e-10

    # for compas
    if data_name == 'compas':
        bs = 64
        lr = .05
        alphas = [.01, .1]
        which_position = 5
    elif data_name == 'adult':
        bs = 256
        lr = .001
        alphas = [.01, .01]
        which_position = 8

    nodes = BaseNodes(data_name, num_nodes, bs, classes_per_node, fairfed=False)
    num_features = len(nodes.features)

    for i in range(num_nodes):

        if fairness == 'none':
            fair = 'none'
        if fairness == 'dp':
            fair = 'dp'
            alpha = alphas[0]
        elif fairness == 'eo':
            fair = 'eo'
            alpha = alphas[1]
        elif fairness == 'both':
            if i % 2 == 0:
                fair = 'dp'
                alpha = alphas[0]
            else:
                fair = 'eo'
                alpha = alphas[1]

        print("\nTraining Client: ", i+1, fair, alpha)
        train(device=device, steps=num_steps, lr=lr, wd=wd, alpha=alpha, fair=fair, which_position=which_position, num_features=num_features, train_loader=nodes.train_loaders[i], test_loader=nodes.test_loaders[i], client_num=i+1)

if __name__ == "__main__":
    main()