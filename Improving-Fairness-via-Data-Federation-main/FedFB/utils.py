from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import pandas as pd
from scipy.stats import multivariate_normal
import torch, random, copy, os
from sklearn.preprocessing import LabelEncoder

################## MODEL SETTING ########################
DEVICE = 'cuda:5'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################
    
class logReg(torch.nn.Module):
    """
    Logistic regression model.
    """
    def __init__(self, num_features, num_classes, seed = 123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x.float())
        probas = torch.sigmoid(logits)
        return probas.type(torch.FloatTensor), logits

def logit_compute(probas):
    return torch.log(probas/(1-probas))
    
def riskDifference(n_yz, absolute = True):
    """
    Given a dictionary of number of samples in different groups, compute the risk difference.
    |P(Group1, pos) - P(Group2, pos)| = |N(Group1, pos)/N(Group1) - N(Group2, pos)/N(Group2)|
    """
    n_z1 = max(n_yz[(1,1)] + n_yz[(0,1)], 1)
    n_z0 = max(n_yz[(0,0)] + n_yz[(1,0)], 1)
    if absolute:
        return abs(n_yz[(1,1)]/n_z1 - n_yz[(1,0)]/n_z0)
    else:
        return n_yz[(1,1)]/n_z1 - n_yz[(1,0)]/n_z0

def pRule(n_yz):
    """
    Compute the p rule level.
    min(P(Group1, pos)/P(Group2, pos), P(Group2, pos)/P(Group1, pos))
    """
    return min(n_yz[(1,1)]/n_yz[(1,0)], n_yz[(1,0)]/n_yz[(1,1)])

def DPDisparity(n_yz, each_z = False):
    """
    Same metric as FairBatch. Compute the demographic disparity.
    max(|P(pos | Group1) - P(pos)|, |P(pos | Group2) - P(pos)|)
    """
    z_set = sorted(list(set([z for _, z in n_yz.keys()])))
    p_y1_n, p_y1_d, n_z = 0, 0, []
    for z in z_set:
        p_y1_n += n_yz[(1,z)]
        n_z.append(max(n_yz[(1,z)] + n_yz[(0,z)], 1))
        for y in [0,1]:
            p_y1_d += n_yz[(y,z)]
    p_y1 = p_y1_n / p_y1_d

    if not each_z:
        return max([abs(n_yz[(1,z)]/n_z[z] - p_y1) for z in z_set])
    else:
        return [n_yz[(1,z)]/n_z[z] - p_y1 for z in z_set]

def EODisparity(n_eyz, each_z = False):
    """
    Equal opportunity disparity: max_z{|P(yhat=1|z=z,y=1)-P(yhat=1|y=1)|}

    Parameter:
    n_eyz: dictionary. #(yhat=e,y=y,z=z)
    """
    z_set = list(set([z for _,_, z in n_eyz.keys()]))
    if not each_z:
        eod = 0
        p11 = sum([n_eyz[(1,1,z)] for z in z_set]) / sum([n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])
        for z in z_set:
            try:
                eod_z = abs(n_eyz[(1,1,z)]/(n_eyz[(0,1,z)] + n_eyz[(1,1,z)]) - p11)
            except ZeroDivisionError:
                if n_eyz[(1,1,z)] == 0: 
                    eod_z = 0
                else:
                    eod_z = 1
            if eod < eod_z:
                eod = eod_z
        return eod
    else:
        eod = []
        p11 = sum([n_eyz[(1,1,z)] for z in z_set]) / sum([n_eyz[(1,1,z)]+n_eyz[(0,1,z)] for z in z_set])
        for z in z_set:
            try:
                eod_z = n_eyz[(1,1,z)]/(n_eyz[(0,1,z)] + n_eyz[(1,1,z)]) - p11
            except ZeroDivisionError:
                if n_eyz[(1,1,z)] == 0: 
                    eod_z = 0
                else:
                    eod_z = 1
            eod.append(eod_z)
        return eod

def average_weights(w, clients_idx, idx_users):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    num_samples = 0
    for i in range(1, len(w)):
        num_samples += len(clients_idx[idx_users[i]])
        for key in w_avg.keys():            
            w_avg[key] += w[i][key] * len(clients_idx[idx_users[i]])
        
    for key in w_avg.keys(): 
        w_avg[key] = torch.div(w_avg[key], num_samples)
    return w_avg

def weighted_average_weights(w, nc, n):
    w_avg = copy.deepcopy(w[0])
    for i in range(1, len(w)):
        for key in w_avg.keys():            
            w_avg[key] += w[i][key] * nc[i]
        
    for key in w_avg.keys(): 
        w_avg[key] = torch.div(w_avg[key], n)
    return w_avg

def loss_func(option, logits, targets, outputs, sensitive, larg = 1):
    """
    Loss function. 
    """
    acc_loss = F.cross_entropy(logits, targets, reduction = 'sum')
    fair_loss0 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(), logits.T[0] - torch.mean(logits.T[0]))
    fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0)) 
    fair_loss1 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(), logits.T[1] - torch.mean(logits.T[1]))
    fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1)) 
    fair_loss = fair_loss0 + fair_loss1

    if option == 'local zafar':
        return acc_loss + larg*fair_loss, acc_loss, larg*fair_loss
    elif option == 'FB_inference':
        acc_loss = F.cross_entropy(logits, torch.ones(logits.shape[0]).type(torch.LongTensor).to(DEVICE), reduction = 'sum')
        return acc_loss, acc_loss, fair_loss
    else:
        return acc_loss, acc_loss, larg*fair_loss

def weighted_loss(logits, targets, weights, mean = True):
    acc_loss = F.cross_entropy(logits, targets, reduction = 'none')
    if mean:
        weights_sum = weights.sum().item()
        acc_loss = torch.sum(acc_loss * weights.to(DEVICE) / weights_sum)
    else:
        acc_loss = torch.sum(acc_loss * weights.to(DEVICE))
    return acc_loss
    
def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def process_csv(dir_name, filename, label_name, favorable_class, sensitive_attributes, privileged_classes, categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = 'infer', columns = None):

    df = pd.read_csv(os.path.join('/home/ancarey/FairFLHN/Improving-Fairness-via-Data-Federation-main/FedFB', dir_name, filename), delimiter = ',', header = header, na_values = na_values)
    if header == None: df.columns = columns
    df = df[features_to_keep]
    df['salary'] = df.salary.str.rstrip('.').astype('category')
    data = df.drop_duplicates()
    data = data.dropna(how='any', axis=0)
    data.capital_gain = data.capital_gain.astype(int)
    data.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married',
                  'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married', 'not married', 'not married', 'not married'],
                 inplace=True)
    data.replace(['Federal-gov', 'Local-gov', 'State-gov'], ['government', 'government', 'government'],
                 inplace=True)
    data.replace(['Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'Without-pay'],
                 ['private', 'private', 'private', 'private', 'private'], inplace=True)
    encoders = {}

    for col in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                'native_country', 'salary']:
        encoders[col] = LabelEncoder().fit(data[col])
        data.loc[:, col] = encoders[col].transform(data[col])

    return data