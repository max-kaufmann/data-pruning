import math
import os

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import yaml
import wandb
import config

round_float = lambda x: round(x, 3)


def visualise_attack(model, attack, inputs, targets, dataset_classes):

    max_length = 50  #Maximum length of probability lists
    inputs, targets = inputs.to(config.device), targets.to(config.device)

    standard_outputs = model(inputs)
    standard_probs = torch.softmax(standard_outputs, dim=1)
    standard_loss = F.cross_entropy(standard_outputs,
                                    targets,
                                    reduction="none")
    standard_classes = torch.argmax(standard_outputs, dim=1)

    adv_inputs = attack(model, inputs, targets)
    adv_outputs = model(adv_inputs)
    adv_probs = torch.softmax(adv_outputs, dim=1)
    adv_loss = F.cross_entropy(adv_outputs, targets, reduction="none")
    adv_classes = torch.argmax(adv_outputs, dim=1)

    num_images, _, _, _ = inputs.shape

    fig, axs = plt.subplots(nrows=num_images,
                            ncols=2,
                            figsize=(30, 50),
                            constrained_layout=True)

    if num_images == 1:
        axs = np.array([axs])

    for i in range(0, num_images):
        stand_ax = axs[i, 0]
        stand_ax.imshow(tensor_to_image(inputs[i]))
        stand_ax.title.set_text(
            f"Standard prediction: {dataset_classes[standard_classes[i]]} \n Probabilities: {str(list(map(round_float, standard_probs[i].tolist())))[:max_length]} \n Standard loss: {standard_loss[i]}"
        )

        adv_ax = axs[i, 1]
        adv_ax.imshow(tensor_to_image(adv_inputs[i]))
        adv_ax.title.set_text(
            f"Adversarial prediction: {dataset_classes[adv_classes[i]]}\n Probabilities: {str(list(map(round_float, adv_probs[i].tolist())))[:max_length]} \n Adversarial loss: {adv_loss[i]}"
        )

    plt.show()


def tensor_to_image(tensor):
    return tensor.permute(1,2,0).detach().cpu()


def plot_image_batch(batch, display_type="square"):

    if len(batch.shape) <= 3:
        for i in range(0, 4 - len(batch.shape)):
            batch = batch.unsqueeze(0)

    batch = torch.stack([tensor_to_image(x) for x in batch])
    num_images, height, width, num_channels = batch.shape

    if num_channels == 1:
        batch = batch.squeeze(-1)

    if display_type == "line":
        image_width = 10
        image_height = 10
        figsize = (image_width, image_height * num_images)
        fig, axs = plt.subplots(nrows=num_images,
                                ncols=1,
                                figsize=figsize,
                                constrained_layout=True)

        if num_images == 1:
            axs = np.array([axs])

        for i in range(0, num_images):
            axs[i].imshow(batch[i])

    elif display_type == "square":

        square_image_height = 10
        square_image_width = 10

        square_size = math.ceil(math.sqrt(num_images))
        figsize = (square_image_height * square_size,
                   square_image_width * square_size)
        fig, axs = plt.subplots(nrows=square_size,
                                ncols=square_size,
                                figsize=figsize,
                                constrained_layout=True)

        if num_images == 1:
            axs = np.array([axs])

        image_index = 0
        for row_index in range(square_size):
            for column_index in range(square_size):
                if image_index >= num_images:
                    break
                else:
                    axs[row_index, column_index].imshow(batch[image_index])
                image_index += 1

    else:
        raise Exception("Batch display type not supported")


def store_images(stand_imgs, adv_imgs, img_num, args):
    stand = torch.cat(list(stand_imgs), dim=-1)
    adv = torch.cat(list(adv_imgs), dim=-1)
    diff = adv - stand
    img = torch.cat((stand, adv, diff / torch.max(torch.abs(diff))), dim=-2)

    save_image(img, os.path.join(args.image_dir, f"{img_num}.png"))


def results_file_to_df(path):
    return pd.read_json(path, lines=True)


def attach_debugger(port=5678):
    debugpy.listen(port)
    print(f"Waiting for debugger on port {port}...")

    debugpy.wait_for_client()
    print(f"Debugger attached on port {port}")

def wandb_sweep_run_init(args):
    """
    Updates sweep parameters at the start of each run of the sweep.
    Names the run.

    Outputs a wandb table object and 
    "data for table" object which is a list of the hyperparameters
    that we want logged in the table.
    
    """

    with open("./experiments/wandb_sweeps/" + args.dataset + "_config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config) #This initialises the run on the wandb side
    
    """
    During a sweep the current run's sweep parameter key-value pairs are appended
    to the start of the wandb.config dictionary. The next three lines slice that off
    so that we can use it for our own initialisation procedures.
    
    """
    number_of_params = len(config['parameters'])
    param_dict = wandb.config
    param_dict = {param: param_dict[param] for param in param_dict.keys()[:number_of_params]}
    
    run_name = ""
    for param, value in param_dict.items():
        setattr(args, param, value) #This line initialises the run within our code by updating args
        run_name += f"{param} = {value} | "
    
    run.name = run_name[:-2]
    
    param_titles = [param.replace("_"," ").title() for param in param_dict.keys()]
    table = wandb.Table(columns=[*param_titles,"Adversarial Accuracy"])

    return {"Table":table, "data for table": [*param_dict.values()] , "Run":run}