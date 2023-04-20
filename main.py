import argparse
import importlib
import os

import jsonlines
import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

import config
import evaluate
from models.model_normalization import Cifar10Wrapper, ImageNetWrapper, Cifar10WrapperWRN, NormalizationWrapper

#TODO: Ask steven about how to add to PYTHONPATH
def get_parser():
    '""Returns the parser object for the main function. NOTE: All arguments should have nargs=1 (so that they can be ran from an experiment file)"""'

    parser = argparse.ArgumentParser()
    
    #General experiment-running arguments
    parser.add_argument("--architecture",
                        type=str,
                        default="convnextv1",
                        help="This specified the model architecture which is being tested")
    parser.add_argument(
        "--weights",
        type=str,
        default="base",
        help=
        "The path to the weights (stored as a state dictonary in a .pt file) which is to be loaded into the model."
    )
    parser.add_argument(
        "--dataset",
        default="imagenet100",
        help=
        "The dataset which is being evaluated."
    )
    parser.add_argument("--attack",
                        type=str,
                        default="pixel",
                        choices=config.attack_list,
                        help="Name of the attack to evaluate against.")
    parser.add_argument(
        "--log",
        type=str,
        default="./log.txt",
        help="Name of the log file which is used to store the results of the experiment, in the jsonlines format.")
    parser.add_argument("--experiment_name",
                        default="experiment",
                        type=str,
                        help="The name of this experiment, used when logging.")
    parser.add_argument(
        "--image_dir",
        default="./",
        help=
        "If the \"--num_image_batches\" argument has been set, thi is the directory where the images are saved."
    )
    parser.add_argument("--num_image_batches",
                        default=1,
                        type=int,
                        help="If this is set to a value n, the first n batches of (image,adversarially_perturbed_image) pairs are saved to the file specified by the --num_image_batches option") #TODO: Make sure that by default, no image batches are saved.
    parser.add_argument(
        "--check_loss_reduction",
        type=lambda x: x == "true" or x == "True",
        default=True,
        help=
        "If set to true then the proportion of images for which loss decreases after the attack is computed and logged. Mainly used for debugging attacks, costs one extra model evaluation per datapoint."
    )
    parser.add_argument("--cuda_autotune",
                        type=lambda x: x == "true" or x == "True",
                        default=True,
                        help="Wether to set torch.backends.cudnn.benchmark to True or False")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Seed used for both numpy and pytorch. Defaults to random.")
    parser.add_argument(
        "--num_workers",
        default=6,
        type=int,
        help="Number of workers which are used by the  Dataloader objects in the project")
    parser.add_argument("--device",
                        default="cpu",
                        help="Device on wh ich experiments are to be ran")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=10,
                        help="The batch sizes used when evaluating models against attacks.")
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help=
        "If set, evaluation will only run for the specified number of batches. This is useful for debugging."
    )
    
    #General attack arguments
    parser.add_argument(
        "--epsilon",
        type=str,
    default="10",
        help="The maximum perturbation allowed in the attack")
    parser.add_argument(
        "--step_size",
        type=float,
        default=None,
        help="The size of the steps used in the PGD optimisation algorithm. By default, is epsilon/num_steps.")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="The number of steps used in the PGD optimisation algorithm.")
    parser.add_argument(
        "--distance_metric",
        type=str,
        choices=["linf", "l2"],
        default="l2",
        help="The distance metric used for constraining the perturbation. Only affects some attacks (see the Github attack README for more details.")

    #Argument for the fog attack
    parser.add_argument(
        "--fog_wibbledecay",
        type=float,
        default=0.99,
        help=
        "Decay parameter for the fog attack, controls the amount of large-scale structure in the fog"
    )
    #Arguments for the glitch attack
    parser.add_argument("--glitch_num_lines", type=int, default=56,help="The number of vertical lines which are added to the image in the glitch attack")
    parser.add_argument("--glitch_grey_strength", type=float, default=0.4,help="The strength of the greying which is applied to the image in the glitch attack")

    #Arguments for the prison attack
    parser.add_argument("--prison_num_bars", type=int, default=9,help="The number of vertical bars which are added to the image in the prison attack")
    parser.add_argument("--prison_bar_width", type=int, default=6,help="The width of the vertical bars which are added to the image in the prison attack")

    #Arguments for the blur attack
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=21,
        help=
        "The size of the kernel which is used for the gaussian blur attack (must be odd) "
    )
    parser.add_argument(
        "--blur_kernel_sigma",
        type=float,
        default=17,
        help=
        "The standard deviation of the blur applied in the gaussian blur attack"
    )
    parser.add_argument(
        "--blur_interp_kernel_size",
        type=int,
        default=11,
        help=
        "The size of the kernel which is used on the interpolation matrix for the gaussian blur attack "
    )
    parser.add_argument(
        "--blur_interp_kernel_sigma",
        type=float,
        default=18,
        help=
        "The standard deviation of the blur applied on the interpolation matrix for the gaussian blur attack"
    )

    #Arguments for the hsv attack
    parser.add_argument(
        "--hsv_kernel_size",
        type=int,
        default=5,
        help=
        "The size of the kernel which is used for the hsv attack (must be odd) "
    )
    parser.add_argument(
        "--hsv_sigma",
        type=float,
        default=5,
        help=
        "The standard deviation of the gaussian blur applied on the HSV channels of the image"
    )

    #Arguments for the gabor attack
    parser.add_argument(
        "--gabor_kernel_size",
        type=int,
        default=23,
        help="The kernel size of the kernels used in the Gabor attack")
    parser.add_argument(
        "--gabor_sides",
        type=int,
        default=1,
        help=
        "The number of times each kernel is rotated and overlaid in the Gabor attack"
    )
    parser.add_argument(
        "--gabor_weight_density",
        type=float,
        default=0.1,
        help=
        "The density of non-zero matrix entries in the spare matrix used in the Gabor noise"
    )
    parser.add_argument("--gabor_sigma",
                        type=float,
                        default=0.4,
                        help="The sigma parameter of the Gabor Kernel")

    #Arguments for the snow attack
    parser.add_argument(
        "--snow_flake_size",
        type=int,
        default=9,
        help=
        "Size of snowflakes (size of the kernel we convolve with). Must be odd."
    )
    parser.add_argument(
        "--snow_num_layers",
        type=int,
        default=4,
        help="Number of different layers of snow applied to each image")
    parser.add_argument(
        '--snow_grid_size',
        type=int,
        default=20,
        help=
        "Distance between adjacent snowflakes (non-zero matrix entries) in the snow grids."
    )
    parser.add_argument(
        '--snow_init',
        type=float,
        default=0.2,
        help="Used to initalise the optimisaiton variables in the snow attack")
    parser.add_argument(
        '--snow_image_discolour',
        type=float,
        default=0.1,
        help=
        "The amount of discolouration applied to the image during the snow attack"
    )
    parser.add_argument(
        '--snow_normalizing_constant',
        type=int,
        default=4,
        help="The normalisation constant applied to the snow flake grid")
    parser.add_argument('--snow_kernel_size',
                        type=int,
                        default=5,
                        help="The size of the kernel used in the snow attack")
    parser.add_argument(
        '--snow_sigma_range_lower',
        default=5,
        type=int,
        help=
        "The lower bound of the range of the sigma in the gaussian blur for the snow attack"
    )
    parser.add_argument(
        '--snow_sigma_range_upper',
        default=6,
        type=int,
        help=
        "The upper bound of the range of the sigma in the gaussian blur for the snow attack"
    )

    #Arguments for the klotski attack
    parser.add_argument("--klotski_num_blocks",
                        type=int,
                        default=8,
                        help="Number of blocks in the klotski attack")

    #Arguments for the whirlpool attack
    parser.add_argument("--num_whirlpools",
                        type=int,
                        default=16,
                        help="Number of whirlpools applied to each image")
    parser.add_argument("--whirlpool_radius",
                        type=float,
                        default=0.5,
                        help="Radius of whirlpool")
    parser.add_argument("--whirlpool_min_strength",
                        type=float,
                        default=0.5,
                        help="Minimum strength of whirlpool")

    #Arguments for the wood attack
    parser.add_argument("--wood_num_rings",
                        type=int,
                        default=5,
                        help="Number of rings of wood applied to each image")
    parser.add_argument("--wood_noise_resolution",
                        type=int,
                        default=16,
                        help="Resolution of the noise used in the wood attack")
    parser.add_argument("--wood_random_init",type=lambda x: x == "true" or x == "True",default=False,help="Whether to use random initialisation for the wood attack")
    parser.add_argument("--wood_normalising_constant",type=int,default=8,help="The normalising constant used in the wood attack")
    
    #Arguments for the interp attack
    parser.add_argument("--mix_interp_kernel_size",
                        type=int,
                        default=5,
                        help="The size of the kernel used in the mix attack")
    parser.add_argument("--mix_interp_kernel_sigma",
                        type=float,
                        default=5,
                        help="The sigma of the kernel used in the mix attack")

    #Arguments for the elastic attack
    parser.add_argument(
        "--elastic_kernel_size",
        type=int,
        default=5,
        help="The size of the kernel used in the elastic attack. Should be odd."
    )
    parser.add_argument(
        "--elastic_kernel_sigma",
        type=float,
        default=5,
        help="The sigma of the kernel used in the elastic attack")

    #Arguments for the pixel attack
    parser.add_argument("--pixel_size",
                        type=int,
                        default=16,
                        help="The size of the pixels in the pixel attack")
    #arguments for the pokadot attack
    parser.add_argument("--pokadot_num_pokadots",
                        type=int,
                        default=20,
                        help="the number of pokadots in the pokadot attack")
    parser.add_argument("--pokadot_distance_scaling",
                        type=float,
                        default=5,
                        help="the distance scaling for the pokadot attack")
    parser.add_argument("--pokadot_image_threshold",
                        type=float,
                        default=0.6,
                        help="the image threshold for the pokadot attack")
    parser.add_argument("--pokadot_distance_normaliser",
                        type=float,
                        default=3,
                        help="the distance normaliser for the pokadot attack")
    

    #Arguments for the kaleidescope attack
    parser.add_argument("--kaleidescope_num_shapes",
                        type=int,
                        default=40,
                        help="The number of shapes in the kaleidescope attack")
    parser.add_argument(
        "--kaleidescope_shape_size",
        type=int,
        default=16,
        help="The size of the shapes in the kaleidescope attack")
    parser.add_argument(
        "--kaleidescope_min_value_valence",
        type=float,
        default=0.8,
        help=
        "The minimum value of the \"value\" parameter in the kaleidescope attack"
    )
    parser.add_argument(
        "--kaleidescope_min_value_saturation",
        type=float,
        default=0.7,
        help=
        "The maximum value of the \"value\" parameter in the kaleidescope attack"
    )
    parser.add_argument(
        "--kaleidescope_transparency",
        type=float,
        default=0.4,
        help=
        "The minimum value of the \"value\" parameter in the kaleidescope attack"
    )
    parser.add_argument(
        "--kaleidescope_edge_width",
        type=int,
        default=3,
        help=
        "The width of the edges in the kaleidescope attack")
    
    #Arguments for the lighting attack
    parser.add_argument("--lighting_num_filters",
                        type=int,
                        default=20,
                        help="The number of filters in the lighting attack")
    parser.add_argument("--lighting_loss_function",
                        type=str,
                        default="cross_entropy",
                        choices=["cross_entropy", "candw"],
                        help="The loss function used in the lighting attack")

    #Arguments for the texture attack
    parser.add_argument(
        "--texture_threshold",
        type=float,
        default=0.1,
        help="The threshold for edges used in the texture attack")

    #Arguments for the edge attack
    parser.add_argument("--edge_threshold",
                        type=float,
                        default=0.1,
                        help="The threshold for edges used in the edge attack")
    #Arguments for the smoke attack
    parser.add_argument("--smoke_resolution",
                        type=int,
                        default=16,
                        help="The resolution of the noise in the smoke attack")
    parser.add_argument(
        "--smoke_freq",
        type=int,
        default=10,
        help="The spacing of the smoke tendrils in the smoke attack")
    parser.add_argument("--smoke_normaliser",
                        type=float,
                        default=4,
                        help="The normaliser for the smoke attack")
    parser.add_argument("--smoke_squiggle_freq",
                        type=float,
                        default=10,
                        help="The squiggle frequency for the smoke attack")
    parser.add_argument("--smoke_height",
                        type=float,
                        default=0.5,
                        help="The height of the smoke in the smoke attack")

    return parser


def run_experiment_file(file, experiment_num=-1):
    """
    Runs a list of experiment from a file, with the file in the jsonlines format, Can either run all experiments in the file
    in sequence, or some specific experiment number within the file/

    Parameters
    ---
    file: str
    jsonlines file, with each line correposnding to ajson object holding the hyperparameters for an experiment

    experiment_num: int
    default -1, which runs all experiments in the file. Otherwise, runs the experiment with the given number.
    """

    experiment_list = []
    with open(file, mode="r") as f:
        reader = jsonlines.Reader(f)

        for experiment in iter(reader):
            parser = get_parser()
            argument_list = []
            for argument, value in experiment.items():
                argument_list.append("--" + str(argument))
                argument_list.append(str(value))

            args = parser.parse_args(argument_list)

            for k, v in vars(args).items():
                if v == "None":
                    vars(args)[k] = None

            experiment_list.append(args)

    if experiment_num >= 0:
        experiment_list = [experiment_list[experiment_num]]

    for experiment in tqdm(iter(experiment_list)):
        main(experiment)


def main(args):
    """
    This function fetches the relevant objects which we need for the evaluation loop, and runs the evaluation loop.
    Throught this function, call by name is used to fetch the objects needed for training based on the input argument strings. In particular:

    1) The dataset is fetched based on the value of the args.dataset argument i.e. models.<args.dataset>.<args.dataset>.get_test_dataloader() is called
    2) The model architecture is fetched based on the architecture parameter and the data parameter i.e. <args.dataset>.<args.architecture>.get_architecture() is called
    3) The attack is fetched based on the attack parameter, i.e. attacks.<args.attack>.get_attack() is called

    """
    init_args(args)  #Do initalisation based on arguments
    print(args)

    model = get_model(
        args.dataset, args.architecture, args.weights, args
    )

    if args.dataset == 'cifar10':
        model = Cifar10WrapperWRN(model) if args.architecture == 'wrn' else Cifar10Wrapper(model)
    elif args.dataset == 'imagenet100':
        model = ImageNetWrapper(model)
    elif args.dataset == 'imagenet':
        if args.architecture == 'vit':
            import timm
            transform = timm.data.create_transform(
                **timm.data.resolve_data_config(model.pretrained_cfg))
            model = NormalizationWrapper(model, mean=transform.transforms[-1].mean,
                                         std=transform.transforms[-1].std)
        else:
            model = ImageNetWrapper(model)
    else:
        raise ValueError('Dataset not supported')


    model.to(args.device)
    model.eval()
    attack = None
    if args.attack != "none":
        attack = get_attack(args.attack, model, args)


    model.eval() 

    test_dataset = get_test_dataset(args.dataset, args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,shuffle=True,batch_size=args.eval_batch_size, num_workers=args.num_workers,pin_memory=True if config.device != "cpu" else False) #
    
    results = evaluate.evaluate(model, test_dataloader, attack, args)
    evaluate.log_csv(results, args.log, args)

    return results


def init_args(args):
    """Does some initilasation based on arguments"""
    if not (args.seed is None):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

    config.device = args.device
    if config.project_path is None:
        config.project_path = os.path.abspath(".")

    if args.cuda_autotune:
        torch.backends.cudnn.benchmark = True

    epsilon_vals =["low","medium","high"] 
    if args.epsilon in epsilon_vals:
        dataset_config = importlib.import_module("models." + args.dataset + "." +
                                             f"{args.dataset}_config",
                                             package=".") 
        args.epsilon = dataset_config.default_epsilons[args.attack][epsilon_vals.index(args.epsilon)]
    else:
        try:
            args.epsilon = float(args.epsilon)
        except ValueError:
            raise ValueError("--epsilon must be given a float or one of the following strings: low, medium, high")
    
    if args.step_size is None:
        args.step_size = args.epsilon/10



def get_model(dataset, architecture, weights, args):
    """Fetches and returns the relevant architecture, based on both the dataset and the architecture name.

    Parameters
    ----
    dataset: string
        dataset from which the architecture is to be fetched

    architecture: string
        name of architecture to be fetched"""

    model_module = importlib.import_module("models." + dataset + "." +
                                             architecture,
                                             package=".")
    model = model_module.get_model( args)
    return model


def get_test_dataset(dataset, args):
    """Fetches the dataloader relevant for the dataset

    Parameters
    ---
    dataset: string
        the name of the dataset from which the dataloader is to be fetched"""
    dataset_module = importlib.import_module("models." + dataset + "." +
                                             dataset,
                                             package=".")
    test_dataset = dataset_module.get_test_dataset(args)
    return test_dataset

def get_train_dataset(dataset, args):
    """Fetches the dataloader relevant for the dataset

    Parameters
    ---
    dataset: string
        the name of the dataset from which the dataloader is to be fetched"""
    dataset_module = importlib.import_module("models." + dataset + "." +
                                             dataset,
                                             package=".")
    train_dataset = dataset_module.get_train_dataset(args)
    return train_dataset

def get_attack(name, model, args):
    """Fetches the attack which needs to be used

    Parameters
    ---
    name: string
       name of the attack which is bein
    model: nn.Module
       model for which the attack is being optimised for, used when constructing the AttackInstance object
    """
    attack_module = importlib.import_module("attacks." + name, package=".")
    attack = attack_module.get_attack(model, args)

    return attack


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
