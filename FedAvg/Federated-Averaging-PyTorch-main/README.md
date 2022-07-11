* NOTE: This repository will be updated to *ver 2.0* at least in August, 2022.
# Federated Averaging (FedAvg) in PyTorch [![arXiv](https://img.shields.io/badge/arXiv-1602.05629-f9f107.svg)](https://arxiv.org/abs/1602.05629)

An unofficial implementation of `FederatedAveraging` (or `FedAvg`) algorithm proposed in the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) in PyTorch. (implemented in Python 3.9.2.)

## Implementation points
* Exactly implement the models ('2NN' and 'CNN' mentioned in the paper) to have the same number of parameters written in the paper.
  * 2NN: `TwoNN` class in `models.py`; 199,210 parameters
  * CNN: `CNN` class in `models.py`; 1,663,370 parameters
* Exactly implement the non-IID data split.
  * Each client has at least two digits in case of using `MNIST` dataset.
* Implement multiprocessing of _client update_ and _client evaluation_.
* Support TensorBoard for log tracking.

## Requirements
* See `requirements.txt`

## Configurations
* See `config.yaml`

## Run
* `python3 main.py`

## Results
### MNIST
* Number of clients: 100 (K = 100)
* Fraction of sampled clients: 0.1 (C = 0.1)
* Number of rounds: 500 (R = 500)
* Number of local epochs: 10 (E = 10)
* Batch size: 10 (B = 10)
* Optimizer: `torch.optim.SGD`
* Criterion: `torch.nn.CrossEntropyLoss`
* Learning rate: 0.01
* Momentum: 0.9
* Initialization: Xavier

Table 1. Final accuracy and the best accuracy 
| Model     | Final Accuracy(IID) (Round) | Best Accuracy(IID) (Round) | Final Accuracy(non-IID) (Round) | Best Accuracy(non-IID) (Round) |
| -----     | -----                       | ----                       | ----                            | ----                           |
| 2NN       | 98.38% (500)                | 98.45% (483)               | 97.50% (500)                    | 97.65% (475)                   |
| CNN       | 99.31% (500)                | 99.34% (197)               | 98.73% (500)                    | 99.28% (493)                   |

Table 2. Final loss and the least loss 
| Model     | Final Loss(IID) (Round) | Least Loss(IID) (Round) | Final Loss(non-IID) (Round) | Least Loss(non-IID) (Round) |
| -----     | -----                   | ----                    | ----                        | ----                        |
| 2NN       | 0.09296 (500)           | 0.06956 (107)           | 0.09075 (500)               | 0.08257 (475)               |
| CNN       | 0.04781 (500)           | 0.02497 (86)            | 0.04533 (500)               | 0.02413 (366)               |

Figure 1. MNIST 2NN model accuracy (IID: top / non-IID: bottom)
![iidmnist](https://user-images.githubusercontent.com/33894768/117546686-95b8c880-b066-11eb-817c-e878a338d28e.png)
![run-Accuracy_ MNIST _TwoNN C_0 1, E_10, B_10, IID_False-tag-Accuracy](https://user-images.githubusercontent.com/33894768/117534148-34bfcf00-b02b-11eb-9b2d-f9a33d05242e.png)

Figure 2. MNIST CNN model accuracy (IID: top / non-IID: bottom)
![run-Accuracy_ MNIST _CNN C_0 1, E_10, B_10, IID_True-tag-Accuracy](https://user-images.githubusercontent.com/33894768/117534156-3b4e4680-b02b-11eb-9f27-ce4a10e7cd6b.png)
![Accuracy](https://user-images.githubusercontent.com/33894768/117542232-c2fb7b80-b052-11eb-90c6-725c94fe0109.png)


## TODO
- [ ] Do CIFAR experiment (CIFAR10 dataset) & large-scale LSTM experiment (Shakespeare dataset)
- [ ] Learning rate scheduling
- [ ] More experiments with other hyperparameter settings (e.g., different combinations of B, E, K, and C)
