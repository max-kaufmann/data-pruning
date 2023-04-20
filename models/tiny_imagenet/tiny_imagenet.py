import importlib

import torch
import torchvision
import torchvision.transforms as transforms

import config as main_config
import models.tiny_imagenet.tiny_imagenet_config as tiny_imagenet_config

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=tiny_imagenet_config.mean,std=tiny_imagenet_config.std)])

def get_test_dataset(args):
    test_dataset = torchvision.datasets.ImageNet(main_config.project_path + "models/tiny_imagenet/data/tiny-imagenet-200",split="train",transform=test_transform)
    return test_dataset





