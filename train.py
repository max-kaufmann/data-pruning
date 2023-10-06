import argparse
import importlib
import numpy as np
import torch
import yaml
import random
import config
from src.misc import attach_debugger
from src.training_loop import train
from evaluate import evaluate

global wandb

def main(args):

    if args.wandb_sweep:
        with open("./experiments/wandb_sweeps/config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        run = wandb.init(config=config)

        args.data_proportion = wandb.config.data_proportion
        #args.pruning_epoch = wandb.config.pruning_epoch
        #args.lr_max = wandb.config.lr_max
        #args.num_epochs = wandb.config.num_epochs
        #args.epsilon = wandb.config.epsilon
        #args.step_size = wandb.config.step_size

        run.name = f"p = {args.data_proportion}"

        table = wandb.Table(columns=["data proportion","adversarial accuracy"])

    elif not args.no_wandb:
        wandb.init(project=args.wandb_project_name, name=args.experiment_name, config=args)

    train_dataset = get_train_dataset(args.dataset,args)
    eval_dataset = get_test_dataset(args.dataset,args)

    model = get_model(args.dataset,args.architecture,args)

    train_attack_args = argparse.Namespace(
        epsilon=args.epsilon,
        step_size=args.step_size,
        num_steps=args.num_steps,
        distance_metric=args.distance_metric)

    train_attack = get_attack(args.attack,train_attack_args)

    eval_attack_args = argparse.Namespace(
        epsilon=args.eval_epsilon,
        step_size=args.eval_step_size,
        num_steps=args.eval_num_steps,
        distance_metric=args.eval_distance_metric)
    
    eval_attack = get_attack(args.eval_attack,eval_attack_args)

    if args.eval_only is not None:
        model.load_state_dict(torch.load(args.eval_only))
        dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        adv_accuracy = evaluate(model, dataloader, eval_attack, args)["test_accuracy"]
        print(f'Attack: {args.eval_attack} | Adversarial Accuracy: {"%.3f" % adv_accuracy}')
    else:
        adv_accuracy = train(model, train_dataset, eval_dataset, train_attack, eval_attack, args)["adv_accuracy"]

    if args.wandb_sweep:
        table.add_data(args.data_proportion,adv_accuracy + random.uniform(-0.001,0.001))
        run.log({"Test Table": table})
    
    if args.save_model is not None:
        torch.save(model.state_dict(), args.save_model)


def get_parser():
    """Returns the parser object for the main function"""   

    parser = argparse.ArgumentParser()

    # General experiment-running arguments
    
    parser.add_argument("--architecture",
                        type=str,
                        default="cnn",
                        help="This specified the model architecture which is being tested")
    
    parser.add_argument("--arch_variant",
                        type=str,
                        default=None,
                        help="This specifies the architecture variant which is being tested. See 'architecture.py' file for options.")

    parser.add_argument("--dataset",
                        default="mnist",
                        help="The dataset which is being evaluated."
                        )
    
    parser.add_argument("--save_model",
                        type=str,
                        default=None,
                        help="Save the model to the given path."
                        )

    parser.add_argument("--pruning_epoch",
                        type=int,
                        default=5,
                        help="The epoch at which to prune the model")

    parser.add_argument("--num_epochs",
                        type=int,
                        default=100,
                        help="The number of epochs to train for")

    parser.add_argument("--pruning_method",
                        type=str,
                        default="high",
                        choices=config.pruning_methods)
    
    parser.add_argument("--systematic_sampling",
                        action="store_true",
                        help="Whether to use systematic sampling or not",
                        )

    parser.add_argument("--attack",
                        type=str,
                        default="fgsm",
                        choices=config.attack_list,
                        help="Name of the attack to evaluate against.")

    parser.add_argument("--lr_max",
                        type=float,
                        default=0.2,
                        help="The maximum learning rate to use for training")

    parser.add_argument("--epsilon",
                        type=float,
                        default=4/255,
                        help="The maximum perturbation allowed in the attack")

    parser.add_argument("--lr_warmup_end",
                        type=float,
                        default=0.4,
                        help="The number of epochs to warm up the learning rate for")

    parser.add_argument("--step_size",
                        type=float,
                        default=None,
                        help="The size of the steps used in the PGD optimization algorithm. By default, is epsilon/num_steps.")

    parser.add_argument("--num_steps",
                        type=int,
                        default=1,
                        help="The number of steps used in the PGD optimization algorithm.")

    parser.add_argument("--distance_metric",
                        type=str,
                        choices=["linf", "l2"],
                        default="linf",
                        help="The distance metric used for constraining the perturbation. Only affects some attacks (see the Github attack README for more details.")

    parser.add_argument("--eval_only",
                    type=str,
                    default=None,
                    help="Whether or not to load in a pretrained model from the given path for evaluation only.")

    parser.add_argument("--eval_attack",
                        type=str,
                        default="pgd",
                        choices=config.attack_list,
                        help="Name of the attack to evaluate against.")

    parser.add_argument("--eval_num_steps",
                        type=int,
                        default=40,
                        help="The number of steps to run the evaluation attack for.")

    parser.add_argument("--eval_epsilon",
                        type=float,
                        default=None,
                        help="The maximum perturbation allowed in the attack. By default the same as the training attack.")

    parser.add_argument("--eval_step_size",
                        type=float,
                        default=None,
                        help="The size of the steps used in the PGD optimization algorithm. By default, is epsilon/num_steps.")

    parser.add_argument("--eval_distance_metric",
                        type=str,
                        choices=["linf", "l2"],
                        default="linf",
                        help="The distance metric used for constraining the perturbation. Only affects some attacks (see the Github attack README for more details.")

    parser.add_argument("--experiment_name",
                        default="experiment",
                        type=str,
                        help="The name of this experiment, used when logging.")

    parser.add_argument("--num_logs_per_epoch",
                        type=int,
                        default=1,
                        help="The number of times to log per epoch.")
    
    parser.add_argument("--lr",
                        type=float,
                        default=0.1,
                        help="The learning rate used for training.")

    parser.add_argument("--cuda_autotune",
                        type=lambda x: x == "true" or x == "True",
                        default=True,
                        help="Whether to set torch.backends.cudnn.benchmark to True or False")

    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Seed used for both numpy and pytorch. Defaults to random.")

    parser.add_argument("--num_workers",
                        default=6,
                        type=int,
                        help="Number of workers which are used by the Dataloader objects in the project")

    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device on which experiments are to be ran")
    
    parser.add_argument("--batch_size",
                        default=128,
                        type=int,
                        help="Batch size used for training")

    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=10,
                        help="The batch sizes used when evaluating models against attacks.")

    parser.add_argument("--eval_size",
                        type=int,
                        default=-1,
                        help="If set, evaluation will only run for the specified number of batches. This is useful for debugging."
                        )

    # General attack arguments

    parser.add_argument("--debug",
                        action="store_true",
                        help="If set, the code will run in debug mode, and will attach to a process specified in the --debug_pid argument."
                        )

    parser.add_argument("--debug_port",
                        type=int,
                        default=5768,
                        help="The process ID of the process to attach to when running in debug mode."
                        )
    
    parser.add_argument("--data_proportion",
                        type=float,
                        default=1.0,
                        help="The proportion of the dataset to use for training after pruning.")      

    parser.add_argument("--validation_dataset_proportion",
                        type=float,
                        default=1.0,
                        help="The proportion of the test dataset to use for validation.")

    parser.add_argument("--wandb_project_name",
                        type=str,
                        default="",
                        help="The name of the wandb project to log to. If not set, wandb logging is disabled."
                        )
    
    parser.add_argument("--no-wandb",
                        action="store_true",
                        help="If set, wandb logging is disabled.")
    
    parser.add_argument("--wandb_sweep",
                        action="store_true",
                        help="Whether or not a wandb sweep is happening.",
                        )
    
    """ 
    When wandb_sweep = True, main knows to initialise the sweep hyperparameters properly,
    instead of just taking the values specified by this parser.

    """

    parser.add_argument("--early_stopping",
                        action="store_true",
                        help="If set, early stopping is used to terminate training early.")

    parser.add_argument("--early_stopping_patience",
                        type=int,
                        default=5,
                        help="The number of epochs to wait before early stopping.")

    return parser


def init_args(args):
    """Does some initialization based on arguments"""
    config.device = args.device

    if not (args.seed is None):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.cuda_autotune:
        torch.backends.cudnn.benchmark = True
    
    if args.step_size is None:
        args.step_size = args.epsilon / args.num_steps
    
    if args.eval_epsilon is None:
        args.eval_epsilon = args.epsilon
    
    if args.eval_step_size is None:
        args.eval_step_size = args.epsilon / args.eval_num_steps

    if not args.no_wandb:
        global wandb
        wandb = importlib.import_module("wandb")
    

def get_model(dataset, architecture, args):
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
    model = model_module.get_model(args)

    return model


def get_test_dataset(dataset, args):
    """Fetches the dataloader relevant for the dataset

    Parameters
    ---
    dataset: string
        the name of the dataset from which the dataloader is to be fetched"""

    dataset_module = importlib.import_module("project_datasets." + dataset + "." +
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

    dataset_module = importlib.import_module("project_datasets." + dataset + "." +
                                             dataset,
                                             package=".")
    train_dataset = dataset_module.get_train_dataset(args)

    return train_dataset


def get_attack(name, args):
    """Fetches the attack which needs to be used

    Parameters
    ---
    name: string
       name of the attack which is being fetched
    model: nn.Module
       model for which the attack is being optimised for, used when constructing the AttackInstance object
    """

    attack_module = importlib.import_module("attacks." + name, package=".")
    attack = attack_module.get_attack(args)

    return attack


if __name__ == "__main__":

    args = get_parser().parse_args()

    if args.debug:
        attach_debugger(args.debug_port)

    init_args(args)

    main(args)
