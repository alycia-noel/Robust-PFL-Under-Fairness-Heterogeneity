# load modules and dataset
from DP_server import *
from DP_load_dataset import *

def run_dp(method, model, dataset, prn = True, seed = 123, trial = False, **kwargs):
    arc = logReg

    if dataset == 'adult':
        Z, num_features, info = 2, adult_num_features, adult_info
    elif dataset == 'compas':
        Z, num_features, info = compas_z, compas_num_features, compas_info

    # set up the server
    server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed, Z = Z, ret = True, prn = prn, trial = trial)

    # execute
    if method == 'fedavg':
        acc, dpdisp, classifier = server.FedAvg(**kwargs)
    elif method == 'uflfb':
        acc, dpdisp, classifier = server.UFLFB(**kwargs)
    elif method == 'fedfb':
        acc, dpdisp, classifier = server.FedFB(**kwargs)
    elif method == 'fflfb':
        acc, dpdisp, classifier = server.FFLFB(**kwargs)

    if not trial: return {'accuracy': acc, 'DP Disp': dpdisp}

def main():
    run_dp('fedfb', 'logistic regression', 'adult', prn=True, seed=123, trial=False)

if __name__ == "__main__":
    main()