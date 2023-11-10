
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
        self.data.targets = self.targets[mask]

    def class_dist(self):
        num_classes = np.max(self.data.targets) + 1
        class_count = torch.tensor([None]*num_classes)
        for i in range(0,num_classes):
            class_count[i] = sum(self.data.targets == i)
        return class_count/sum(class_count)


        