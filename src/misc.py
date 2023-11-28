import math
import os

import debugpy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import wandb
import config
import copy
import torch
import torch.nn.functional as F
from torch.linalg import norm
from torch.autograd import Variable
from torchvision.utils import save_image

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

        image_index = 0
        if num_images == 1:
            axs.imshow(batch[image_index]) 
        else:
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

def reset_wandb_env():
    """Workaround helper function to solve the wandb.init() overwriting problem:
    When you call wandb.sweep then wandb.agent and then (after the sweep finishes) wandb.init
    all from the same script, the run created by wandb.init() will overwrite the last run of 
    the sweep, even if wandb.finish is called before wandb.init(). This function resets the 
    wandb env variables, forcing a new run to be started."""
    for key in os.environ.keys():
        if key.startswith("WANDB_"):
            del os.environ[key]

def wandb_sweep_run_init(args):
    """
    Updates sweep parameters at the start of each run of the sweep.
    Names the run.

    Outputs a wandb table object and 
    "data for table" object which is a list of the hyperparameters
    that we want logged in the table.
    
    """
    run = wandb.init() #This initialises the run on the wandb side

    param_dict = dict(wandb.config)
    run_name = ""
    for param, value in param_dict.items():
        if param not in args and param != "repeat":
            raise AttributeError(f"args does not have the attribute '{param}'")
        setattr(args, param, value) #This line initialises the run within our code by updating args
        run_name += f"__{param}_{value}"
    
    run.name = wandb.config._settings.sweep_id + run_name
    
    param_titles = [param.replace("_"," ").title() for param in param_dict.keys()]
    dataframe = pd.DataFrame(columns=[*param_titles,"Class Distribution","Adversarial Accuracy"])

    return {"DataFrame":dataframe, "data for table": [*param_dict.values()] , "Run":run}

def deepfool(images, model, args, num_classes=10, overshoot=0.02, max_iter=5):

    """
       :param image: Image of size HxWx3
       :param model: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    images = images.to(args.device)
    model = model.to(args.device)
    
    model_is_training = model.training
    if model_is_training:
        model.eval()
    
    #manage indices
    f = model(images)
    I = f.argsort(dim=1,descending=True)
    I = I[:,0:num_classes]

    batch_size = len(I)
    distances = torch.zeros(batch_size)
    
    for n in range(batch_size):

        #initialise variables
        label = I[n,0]
        x = copy.deepcopy(images[n])
        x = x.unsqueeze(0)
        x = x.to(args.device)
        x.requires_grad = True
        w = torch.zeros_like(x)
        r_tot = torch.zeros_like(x)
        loop_i = 0
        logits = model(x)
        k_i = label

        while k_i == label and loop_i < max_iter:

            pert = np.inf
            J = torch.autograd.functional.jacobian(model,x).squeeze()
            #is there a way of getting J and logits without two foward passes (the one in `logits = model(x)` and the one in the line above)?

            for k in range(1, num_classes):
                w_k = J[I[n,k]] - J[label]
                f_k = logits[0,I[n,k]] - logits[0,label]
                pert_k = abs(f_k) / norm(w_k)

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i =  (pert) * w / norm(w)
            r_tot = r_tot + r_i

            x = images[n] + (1+overshoot)*r_tot 
            x = x.detach().to(args.device)
            x.requires_grad = True 
            logits = model(x)
            k_i = logits.argmax()

            loop_i += 1

        distances[n] = norm(r_tot)

    if model_is_training:
        model.train()

    return distances.detach()