import time

import jsonlines
import torch 
import torch.nn.functional as F
import wandb


def evaluate(model, dataloader, attack, args):
    """Runs the evaluation loop, calculating and logging adversarial loss and accuracy. The dataloader here is typically an attacks.attacks.AdversarialDataloader object,
    which generates the relevant adversarial examples."""

    model_mode = model.training
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0
    

    for i, (xs, ys) in enumerate(dataloader):

        xs, ys = xs.to(args.device), ys.to(args.device)

        if attack is not None:
            adv_xs = attack.generate_attack(model,xs, ys)
        else:
            adv_xs = xs

        output = model(adv_xs)
        adv_loss = F.cross_entropy(output, ys,reduction="none")
        total_loss += adv_loss.sum().item()


        total_correct += torch.sum(torch.argmax(output, dim=1) == ys)
        total_samples += len(ys)
        

    accuracy = total_correct / total_samples
    loss = total_loss / total_samples

    metrics = {"test_loss": loss, "test_accuracy": accuracy.item()}
    
    

    if model_mode:
        model.train()


    return metrics


def log_csv(metrics, filename, args=None):
    if args is not None:
        args = vars(args)  # Converts the args Namespace to a dictionary
    else:
        args = {}

    dictonary = dict(metrics, **args)  # Merge the two dictionaries

    with open(filename, mode='a+') as f:
        writer = jsonlines.Writer(f)
        writer.write(dictonary)
