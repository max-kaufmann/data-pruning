import torch
import torch.nn as nn
import torchvision.transforms as transforms

from project_datasets.mnist.mnist_config import mn,std

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mnist_net():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def get_model(args):

    model = mnist_net()

    model_normalized = nn.Sequential(transforms.Normalize(mn,std),model) #TODO: CHECK THESE DON'T CHANGE

    return  model_normalized