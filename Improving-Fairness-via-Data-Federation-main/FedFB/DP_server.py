import torch, copy, time, random, warnings, os
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from ray import tune
import torch.nn as nn

################## MODEL SETTING ########################
DEVICE = 'cuda:5'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################

class TabularData(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        n, m = X.shape
        self.n = n
        self.m = m
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

class Server(object):
    def __init__(self, model, dataset_info, seed = 123, num_workers = 4, ret = False, 
                train_prn = False, metric = "Demographic disparity", 
                batch_size = 128, print_every = 1, fraction_clients = 1, Z = 2, prn = True, trial = False):
        """"
        dataset_info: a list of three objects.
            - train_dataset: Dataset object.
            - test_dataset: Dataset object.
            - clients_idx: a list of lists, with each sublist contains the indexs of the training samples in one client.
                    the length of the list is the number of clients.

        """

        self.model = model

        self.model.to(DEVICE)

        self.seed = seed
        self.num_workers = 1

        self.ret = ret  # return the accuracy and fairness measure and print nothins else print the log and return None
        self.prn = prn
        self.train_prn = False if ret else train_prn #if true print the batch loss in local epochs

        #self.metric == "Demographic disparity"
        #self.disparity = DPDisparity

        self.metric = "Equal Opportunity"
        self.disparity = EODisparity

        self.batch_size = batch_size
        self.print_every = print_every
        self.fraction_clients = fraction_clients

        self.train_dataset, self.test_dataset, self.splits = dataset_info

        self.num_clients = 4
        self.Z = Z

        self.trial = trial

        self.train_loaders, self.test_loaders, self.features, self.labels = self.gen_random_loaders(self.train_dataset, self.test_dataset, self.splits, self.batch_size, self.num_clients)

    def gen_random_loaders(self, train, test, splits, bz, num_clients):
        loader_params = {"batch_size": bz, "shuffle": True, "pin_memory": True, "num_workers": 0}

        cols = train.columns
        features, labels = cols[:-1], cols[-1]

        datasets = [train, test]
        dataloaders = []

        all_client_test_train = [[], []]

        first_set = random.randint(1, num_clients - 1)
        second_set = abs(num_clients - first_set)

        assert first_set > 0
        assert second_set > 0

        for j, data_copy in enumerate(datasets):
            amount_two = int((data_copy.shape[0] - splits[j]) / second_set)

            for i in range(num_clients):
                if i < first_set:
                    amount = int((splits[j]) / first_set)
                else:
                    amount = amount_two

                # last client gets the remainders
                if i == num_clients - 1:
                    client_data = data_copy
                else:
                    client_data = data_copy[0:amount + 1]

                all_client_test_train[j].append(TabularData(client_data[features].values, client_data[labels].values))

                data_copy = data_copy[amount + 1:]

            subsets = all_client_test_train[j]
            if j == 0:
                loader_params['shuffle'] = True
            else:
                loader_params['shuffle'] = False

            dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

        return dataloaders[0], dataloaders[1], features, labels


    def FedFB(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3):
        # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()

        # the number of samples whose label is y and sensitive attribute is z - for entire dataset
        m_yz, lbd = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                m_yz[(y,z)] = ((self.train_dataset.salary == y) & (self.train_dataset.sex == z)).sum()

        for y in [0,1]:   # lbd is the percent of the group for each label. I.e., probability that the label is 0 given sens attr 0
            for z in range(self.Z):
                lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses, nc = [], [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_loaders[idx],
                                     test_dataset=self.test_loaders[idx],
                                     batch_size = self.batch_size,
                                     option = "FB-Variant1",
                                     seed = self.seed,
                                     prn = self.train_prn,
                                     Z = self.Z,
                                     feature_col = self.features,
                                     labels_col = self.labels)

                w, loss, nc_ = local_model.fb_update(
                                model=copy.deepcopy(self.model), global_round=round_,
                                    learning_rate = learning_rate / (round_+1), local_epochs = local_epochs,
                                    optimizer = optimizer, lbd = lbd, m_yz = m_yz)
                nc.append(nc_)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = weighted_average_weights(local_weights, nc, sum(nc))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz, loss_yz = {}, {}
            n_eyz, loss_eyz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0
                    loss_yz[(y,z)] = 0

            for e in [0,1]:
                for y in [0,1]:
                    for z in range(self.Z):
                        n_eyz[(e, y, z)] = 0
                        loss_eyz[(e, y, z)] = 0

            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_loaders[c], test_dataset=self.test_loaders[c], batch_size = self.batch_size, option = "FB-Variant1",
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)
                acc, loss, n_yz_c, acc_loss, fair_loss, loss_yz_c, n_eyz_c, loss_eyz_c = local_model.inference(model = self.model)
                list_acc.append(acc)

                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    loss_yz[yz] += loss_yz_c[yz]

                for eyz in n_eyz:
                    n_eyz[eyz] += n_eyz_c[eyz]
                    loss_eyz[eyz] += loss_eyz_c[eyz]

                if self.disparity == DPDisparity:
                    if self.prn: print("Client %d: accuracy loss: %.4f | fairness loss %.4f | %s = %.4f" % (
                        c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))
                else:
                    print("Client %d: accuracy loss: %.4f | fairness loss %.4f | %s = %.4f" % (c+1, acc_loss, fair_loss, self.metric, self.disparity(n_eyz_c)))

            # update the lambda according to the paper -> see Section A.1 of FairBatch
            for y, z in loss_yz:
                loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
            y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
            if y0_diff > y1_diff:
                lbd[(0,0)] -= alpha / (round_+1)
                lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                lbd[(1,0)] = 1 - lbd[(0,0)]
                lbd[(0,1)] += alpha / (round_+1)
                lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                lbd[(1,1)] = 1 - lbd[(0,1)]
            else:
                lbd[(0,0)] += alpha / (round_+1)
                lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                lbd[(0,1)] = 1 - lbd[(0,0)]
                lbd[(1,0)] -= alpha / (round_+1)
                lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                lbd[(1,1)] = 1 - lbd[(1,0)]

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.disparity == DPDisparity:
                if self.prn:
                    if (round_ + 1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_ + 1} global rounds:')
                        print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                            np.mean(np.array(train_loss)),
                            100 * train_accuracy[-1], self.metric, self.disparity(n_yz)))
            else:
                if self.prn:
                    if (round_ + 1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_ + 1} global rounds:')
                        print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                            np.mean(np.array(train_loss)),
                            100 * train_accuracy[-1], self.metric, self.disparity(n_eyz)))




        # Test inference after completion of training
        test_acc, n_yz, n_eyz = self.test_inference(self.model, self.test_dataset)

        if self.disparity == DPDisparity:
            rd = self.disparity(n_yz)
        else:
            rd = self.disparity(n_eyz)

        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        if self.ret: return test_acc, rd, self.model

    def test_inference(self, model = None, test_dataset = None):

        """
        Returns the test accuracy and fairness level.
        """
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        DEVICE = 'cuda:5'

        if model == None: model = self.model
        test_dataset = self.test_dataset
        model.eval()
        total, correct = 0.0, 0.0
        n_yz = {}
        for y in [0,1]:
            for z in range(self.Z):
                n_yz[(y,z)] = 0

        n_eyz = {}
        for e in [0,1]:
            for y in [0, 1]:
                for z in range(self.Z):
                    n_eyz[(e, y, z)] = 0

        testloader = DataLoader(TabularData(test_dataset[self.features].values, test_dataset[self.labels].values), batch_size=self.batch_size,
                                shuffle=False)

        for _, (features, labels) in enumerate(testloader):
            features = features.to(DEVICE)
            sensitive = features[:, 8].type(torch.LongTensor).to(DEVICE)
            labels =  labels.type(torch.LongTensor).to(DEVICE)
            # Inference
            outputs, _ = model(features)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)

            for y,z in n_yz:
                n_yz[(y,z)] += torch.sum((sensitive == z) & (pred_labels == y)).item()

            for e,y,z in n_eyz:
                n_eyz[(e,y,z)] += torch.sum((sensitive == z) & (pred_labels == e) & (labels == y)).item()

            ## We can call our TP, FP, TN, FN metrics here and then call for SPD and EOD

        accuracy = correct/total

        return accuracy, n_yz, n_eyz

class Client(object):
    def __init__(self, dataset, test_dataset, batch_size, option, seed = 0, prn = True, penalty = 500, Z = 2, feature_col = None, labels_col = None):
        self.seed = seed 
        self.option = option
        self.prn = prn
        self.Z = Z
        self.trainloader = dataset
        self.validloader = test_dataset
        self.penalty = penalty
        self.disparity = DPDisparity
        self.features = feature_col
        self.labels = labels_col

    def fb_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_yz):
        # Set mode to train model
        model.train()
        epoch_loss = []
        nc = 0

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, batch in enumerate(self.trainloader):
                feat, labels = batch
                feat, labels = feat.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                s = feat[:, 8].to(DEVICE)
                _, logits = model(feat)

                logits = logits.to(DEVICE)
                v = torch.randn(len(labels)).type(torch.DoubleTensor).to(DEVICE)
                
                group_idx = {}
                
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (s == z))[0]
                    v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]
                    nc += v[group_idx[(y,z)]].sum().item()

                # print(logits)
                loss = weighted_loss(logits, labels, v)
                # if global_round == 1: print(loss)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(feat),
                        len(self.trainloader),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc


    def inference(self, model, train = False):
        """ 
        Returns the inference accuracy, 
                                loss, 
                                N(sensitive group, pos), 
                                N(non-sensitive group, pos), 
                                N(sensitive group),
                                N(non-sensitive group),
                                acc_loss,
                                fair_loss
        """

        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_yz, loss_yz = {}, {}
        n_eyz, loss_eyz = {}, {}

        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0

        for e in [0,1]:
            for y in [0,1]:
                for z in range(self.Z):
                    loss_eyz[(e,y,z)] = 0
                    n_eyz[(e,y,z)] = 0


        dataset = self.validloader if not train else self.trainloader
        for _, (feat, labels) in enumerate(dataset):
            feat, labels = feat.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
            sensitive = feat[:, 8].type(torch.LongTensor).to(DEVICE)
            
            # Inference
            outputs, logits = model(feat)
            outputs, logits = outputs.to(DEVICE), logits.to(DEVICE)

            # Prediction
            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1

            group_boolean_idx = {}
            group_boolean_idx_eo = {}

            for yz in n_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()     
                
                if self.option in["FairBatch", "FB-Variant1"]:
                # the objective function have no lagrangian term

                    loss_yz_,_,_ = loss_func("FB_inference", logits[group_boolean_idx[yz]].to(DEVICE), 
                                                    labels[group_boolean_idx[yz]].to(DEVICE), 
                                         outputs[group_boolean_idx[yz]].to(DEVICE), sensitive[group_boolean_idx[yz]].to(DEVICE), 
                                         self.penalty)
                    loss_yz[yz] += loss_yz_

            for eyz in n_eyz:
                group_boolean_idx_eo[eyz] = (labels == eyz[1]) & (sensitive == eyz[2]) & (pred_labels == eyz[0])
                n_eyz[eyz] += torch.sum((pred_labels == eyz[0]) & (sensitive == eyz[2]) & (labels == eyz[1])).item()
                if self.option in["FairBatch", "FB-Variant1"]:
                # the objective function have no lagrangian term

                    loss_eyz_,_,_ = loss_func("FB_inference", logits[group_boolean_idx_eo[eyz]].to(DEVICE),
                                                    labels[group_boolean_idx_eo[eyz]].to(DEVICE),
                                         outputs[group_boolean_idx_eo[eyz]].to(DEVICE), sensitive[group_boolean_idx_eo[eyz]].to(DEVICE),
                                         self.penalty)
                    loss_eyz[eyz] += loss_eyz_

            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits, 
                                                        labels, outputs, sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total

        return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, loss_yz, n_eyz, loss_eyz

