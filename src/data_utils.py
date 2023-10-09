
import numpy as np 
import torch

class ShuffledDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = torch.randperm(len(self.dataset))

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.dataset)

    def get_indices(self):
        return self.indices

class PrunableDataset(torch.utils.data.Dataset):
    
    def __init__(self,dataset):
        self.data = dataset
        self.targets = np.array(self.data.targets)

    def __getitem__(self,index):
        data,target = self.data[index]

        return data,target


    def __len__(self):
        return len(self.data)

    def remove_indices(self,indices):
        mask = np.ones(shape=len(self.data),dtype=bool)
        mask[indices] = False
        self.data.data = self.data.data[mask]
        print(f'Targets: {self.data.targets}')
        print(f'Mask: {mask}')
        self.data.targets = self.data.targets[mask]
        