#adapted from from github.com/1M50RRY/resnet18-preact

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import config
from torchvision.models.resnet import conv3x3, _resnet
from project_datasets.cifar10.cifar10_config import mean,std

class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(PreactBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
    
def get_model(args):
    model = _resnet(block = PreactBasicBlock, layers = [2,2,2,2], weights = None, progress = True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model_normalized = nn.Sequential(transforms.Normalize(mean,std),model) #TODO: CHECK THESE DON'T CHANGE

    return  model_normalized