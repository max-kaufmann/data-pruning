import time

import jsonlines
import torch 
import torch.nn.functional as F

import utils


def evaluate(model, dataloader, attack, args):
    """Runs the evaluation loop, calculating and logging adversarial loss and accuracy. The dataloader here is typically an attacks.attacks.AdversarialDataloader object,
    which generates the relevant adversarial examples."""

    model.eval()
    start_time = time.time()

    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    if args.check_loss_reduction:
        total_loss_reduction = 0

    for i, (xs, ys) in enumerate(dataloader):

        xs, ys = xs.to(args.device), ys.to(args.device)

        if attack is not None:
            adv_xs = attack.generate_attack((xs, ys))
        else:
            adv_xs = xs

        output = model(adv_xs)
        adv_loss = F.cross_entropy(output, ys,reduction="none")
        total_loss += adv_loss.sum().item()


        total_correct += torch.sum(torch.argmax(output, dim=1) == ys)
        total_samples += len(ys)
        if args.check_loss_reduction: #Checks if loss has reduced for any of the examples in the batch
            stand_loss = F.cross_entropy(model(xs), ys, reduction="none")
            total_loss_reduction +=  (adv_loss < stand_loss).sum().item()
        

        if args.image_dir is not None and i < args.num_image_batches:
            utils.store_images(xs,adv_xs,i,args)

        
        if i == args.num_batches :  # Early stopping of training
            break


    end_time = time.time()
    total_correct = total_correct.item()

    accuracy = total_correct / total_samples
    loss = total_loss / total_samples
    metrics = {"accuracy": accuracy, "avg_loss": loss, "time": (end_time - start_time)}

    if args.check_loss_reduction:
        metrics["loss_reduction"] = total_loss_reduction / total_samples
    

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
