import torchvision
import torchvision.transforms as transforms

import config


def get_test_dataset(args):
    test_transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.CIFAR10(config.project_path + "/data/datasets/CIFAR10",train=False,download=True,transform=test_transform)
    return test_dataset


def get_train_dataset(args):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.CIFAR10(config.project_path + "/data/datasets/CIFAR10",train=True,download=True,transform=train_transform)
    return train_dataset
