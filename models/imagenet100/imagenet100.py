import os
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import timm

import models.imagenet100.imagenet100_config as imagenet100_config


def get_test_dataset(args):
    # Currently only resnet50 is supported
    assert args.architecture == 'resnet50'
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(imagenet100_config.imagenet100_location, 'val'),transform=test_transform)

    return test_dataset
