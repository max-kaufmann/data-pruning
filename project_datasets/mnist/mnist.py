import torchvision.transforms as transforms
import torchvision.datasets as datasets
from project_datasets.mnist.mnist_config import mnist_dir

def get_test_dataset(args):

    test_dataset =  datasets.MNIST(mnist_dir, train=False, download=True, transform=transforms.ToTensor()) 

    return test_dataset

def get_train_dataset(args):

    train_dataset =  datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms.ToTensor()) 

    return train_dataset


