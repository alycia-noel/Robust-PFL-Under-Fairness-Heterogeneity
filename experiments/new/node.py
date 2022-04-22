from experiments.new.dataset import gen_random_loaders
import torch

class BaseNodes:
    def __init__(self, data_name, n_nodes, batch_size, classes_per_node):
        self.data_name = data_name
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.classes_per_node = classes_per_node
        self.train_loaders, self.test_loaders, self.features = None, None, None
        self.c_i = None
        self._init_dataloaders()

    def _init_dataloaders(self):
        loaders, self.features = gen_random_loaders(self.data_name, self.n_nodes, self.batch_size, self.classes_per_node)
        self.train_loaders, self.test_loaders = loaders
        self.c_i = [torch.rand((1, len(self.features))) for i in range(self.n_nodes)]

    def __len__(self):
        return self.n_nodes