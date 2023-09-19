import os
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import timm

import d.imagenet.imagenet_config as imagenet_config


def get_test_dataset(args):
    if args.architecture == 'vit':
        model = timm.create_model(args.weights, pretrained=True).eval()
        transform = timm.data.create_transform(
            **timm.data.resolve_data_config(model.pretrained_cfg))
        assert isinstance(transform.transforms[-2], transforms.ToTensor)
        assert isinstance(transform.transforms[-1], transforms.Normalize)
        test_transform = transforms.Compose(transform.transforms[:-1])
    else:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    test_dataset = torchvision.datasets.ImageNet(imagenet_config.imagenet_location,split="val",transform=test_transform)

    return test_dataset

def get_train_dataset(args):
    test_dataset = torchvision.datasets.ImageNet(imagenet_config.imagenet_location,split="train",transform=test_transform)
    return test_dataset
