from src.data_utils import ShuffledDataset, PrunableDataset
from src.deepfool import distance
import torch
import pandas as pd
import time
import copy
import wandb
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import math
from evaluate import evaluate
import attacks
import sys
import os


def get_remove_indices(metric_tensor, original_indices, args):
    metric_tensor_indices = torch.argsort(metric_tensor)

    if args.pruning_method == "high":
        shuffled_indices_to_remove = metric_tensor_indices[
            math.floor(args.data_proportion * len(metric_tensor_indices)) :
        ]
        indices_to_remove = original_indices[shuffled_indices_to_remove.cpu()]
    elif args.pruning_method == "low":
        shuffled_indices_to_remove = metric_tensor_indices[
            : math.floor((1 - args.data_proportion) * len(metric_tensor_indices))
        ]
        indices_to_remove = original_indices[shuffled_indices_to_remove.cpu()]
    elif args.pruning_method == "low+high":
        shuffled_indices_to_remove = torch.cat(
            [
                metric_tensor_indices[
                    : math.floor(
                        (1 - args.data_proportion) * len(metric_tensor_indices)
                    )
                    // 2
                ],
                metric_tensor_indices[
                    -math.floor((1 - args.data_proportion) * len(metric_tensor_indices))
                    // 2 :
                ],
            ]
        )
        indices_to_remove = original_indices[shuffled_indices_to_remove.cpu()]
    elif args.pruning_method == "random":
        indices_to_remove = np.random.choice(
            original_indices,
            math.floor((1 - args.data_proportion) * len(original_indices)),
            replace=False,
        )
    else:
        raise ValueError("Pruning method not recognized")

    return indices_to_remove


def smaller_dataloader(dataset, size, args):
    dataset_size = size if size != -1 else len(dataset)
    smaller_dataset = torch.utils.data.Subset(
        dataset, np.random.choice(len(dataset), dataset_size, replace=False)
    )
    return torch.utils.data.DataLoader(
        smaller_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def train(
    model: torch.nn.Module,
    train_dataset,
    eval_dataset,
    optimizer,
    train_attack,
    eval_attack,
    args,
):
    model.to(args.device)
    model.train()

    lr_generator = lambda t: np.interp(t,[0, args.num_epochs * args.lr_warmup_end, args.num_epochs],[0, args.lr_max, 0],)  # TODO: replace with a proper lr scheduler call

    start_time = time.time()

    # To make sure we keep a correspondence between the indices of the dataset and the indices of the dataloader, we shuffle the dataset before

    train_dataset = PrunableDataset(train_dataset)

    # make eval dataset smaller
    eval_dataloader = smaller_dataloader(eval_dataset, args.eval_size, args)

    if args.data_proportion > 0.1 and args.dataset == 'cifar10':
        args.early_stopping = False

    if args.early_stopping:
        early_stopping_dataloader = smaller_dataloader(
            eval_dataset, args.early_stopping_size, args
        )
        best_accuracy = 0
        best_epoch = -1

    for epoch in range(0, args.num_epochs):
        train_dataset_shuffled = ShuffledDataset(train_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_shuffled,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        train_loader_size = len(train_dataloader)

        is_pruning_epoch = epoch + 1 == args.pruning_epoch

        if is_pruning_epoch:
            if args.pruning_metric == "compare":
                loss_list = []
                distance_list = []
            else:
                metric_list = []

        for i, (xs, ys) in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}"):
            lr = lr_generator(epoch + (i + 1) / len(train_dataloader))
            optimizer.param_groups[0].update(lr=lr)

            xs, ys = xs.to(args.device), ys.to(args.device)

            adv_xs = train_attack.generate_attack(model, xs, ys)
            logits = model(adv_xs)

            loss = F.cross_entropy(logits, ys, reduction="none")

            if is_pruning_epoch and args.pruning_method != "random":
                if args.pruning_metric == "compare":
                    loss_list.append(loss)
                    distance_list.append(distance(xs, model, args))
                elif args.pruning_metric == "loss":
                    metric_list.append(loss)
                elif args.pruning_metric == "distance":
                    metric_list.append(distance(xs, model, args))
                else:
                    raise ValueError(
                        f"Pruning metric '{args.pruning_metric}' not supported"
                    )

            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

            if i % math.ceil((train_loader_size - 1) / 20) == 0:
                if not args.no_wandb:
                    wandb.log(
                        {
                            "train_loss": loss,
                            "epoch": epoch + i / train_loader_size,
                            "lr": lr,
                            "dataset_size": len(train_dataset),
                        }
                    )
                else:
                    print(
                        f"Epoch {epoch + i/train_loader_size} - loss: {loss} - lr: {lr}"
                    )

        end_time = int(time.time()) - int(start_time)

        if not args.no_wandb:
            wandb.log({"epoch": epoch, "time": end_time})
        else:
            print(f"Epoch {epoch} - time: {end_time}")

        if args.early_stopping:
            metrics = evaluate(model, early_stopping_dataloader, eval_attack, args)
            if not args.no_wandb:
                wandb.log(metrics)
            else:
                print(metrics)

            if metrics["test_accuracy"] > best_accuracy:
                best_accuracy = metrics["test_accuracy"]
                best_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())

            if epoch - best_epoch >= args.early_stopping_patience:
                print("EARLY STOPPING")
                model.load_state_dict(best_model)
                break

        elif args.num_logs_per_epoch == 1:
            metrics = evaluate(model, eval_dataloader, eval_attack, args)
            if not args.no_wandb:
                wandb.log(metrics)
            else:
                print(metrics)

        if is_pruning_epoch:
            if args.pruning_metric == "compare":
                loss_tensor = torch.cat(loss_list)
                distance_tensor = torch.cat(distance_list)

                data = [[x.item(), y.item()] for (x, y) in zip(distance_tensor, loss_tensor)]
                df = pd.DataFrame(data=data, columns=["Distance", "Loss"])
                os.makedirs(f"./experiments/wandb_sweeps/logs/distance_loss_correlation/", exist_ok=True)
                df.to_json(f"./experiments/wandb_sweeps/logs/distance_loss_correlation/pe_{epoch}.json")
                table = wandb.Table(data=data, columns=["Distance", "Loss"])
                wandb.log(
                    {
                        "Distance vs Loss Scatter": wandb.plot.scatter(
                            table, "Distance", "Loss"
                        )
                    }
                )

                sys.exit()

            elif args.data_proportion != 1:
                if args.pruning_method != "random":
                    metric_tensor = torch.cat(metric_list)
                else:
                    metric_tensor = torch.tensor(metric_list)

                shuffled_index = train_dataset_shuffled.get_indices()

                if args.systematic_sampling:
                    original_targets = train_dataset.targets
                    shuffled_targets = original_targets[shuffled_index]

                    num_classes = np.max(original_targets) + 1

                    list_of_indices_to_remove = []
                    for i in range(0, num_classes):
                        shuffled_targets_class_mask = shuffled_targets == i
                        original_indices = shuffled_index[shuffled_targets_class_mask]
                        metric_tensor_class = metric_tensor[shuffled_targets_class_mask]

                        indices_to_remove = get_remove_indices(
                            metric_tensor_class, original_indices, args
                        )
                        list_of_indices_to_remove.append(indices_to_remove)

                    indices_to_remove = np.concatenate(list_of_indices_to_remove)
                else:
                    indices_to_remove = get_remove_indices(
                        metric_tensor, shuffled_index, args
                    )

                train_dataset.remove_indices(indices_to_remove)

            elif isinstance(train_dataset.data.targets, torch.Tensor):
                #PrunableDataset.remove_indices turns PrunableDataset.data.targets into an np.array (from either a torch.Tensor in the case of MNIST or a list in the case of CIFAR10).
                #If data prop = 1 we dont prune but we still need to convert any torch.Tensors to numpy.arrays in order for PrunableDataset.class_dist() to work.
                train_dataset.data.targets = torch.Tensor.numpy(
                    train_dataset.data.targets
                )

    if args.early_stopping:
        final_accuracy = evaluate(model, eval_dataloader, eval_attack, args)["test_accuracy"]
    elif args.num_logs_per_epoch == 0:
        final_accuracy = evaluate(model, eval_dataloader, eval_attack, args)["test_accuracy"]
    else:
        final_accuracy = metrics["test_accuracy"]

    if not args.no_wandb:
        wandb.log({"adv_accuracy": final_accuracy})
    else:
        print(f"Advesraial accuracy: {final_accuracy}")

    return model, final_accuracy, train_dataset.class_dist()
