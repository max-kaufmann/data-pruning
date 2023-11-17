from src.data_utils import ShuffledDataset, PrunableDataset
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

def get_remove_indices(loss_tensor,original_indices,args):
    
    loss_tensor_indices=torch.argsort(loss_tensor)

    if args.pruning_method == "high":
        shuffled_indices_to_remove = loss_tensor_indices[math.floor(args.data_proportion * len(loss_tensor_indices)):]
        indices_to_remove = original_indices[shuffled_indices_to_remove.cpu()]
    elif args.pruning_method == "low":
        shuffled_indices_to_remove = loss_tensor_indices[:math.floor((1-args.data_proportion) * len(loss_tensor_indices))]
        indices_to_remove = original_indices[shuffled_indices_to_remove.cpu()]
    elif args.pruning_method == "low+high":
        shuffled_indices_to_remove = torch.cat([loss_tensor_indices[:math.floor((1-args.data_proportion) * len(loss_tensor_indices))//2], loss_tensor_indices[-math.floor((1-args.data_proportion) * len(loss_tensor_indices))//2:]])
        indices_to_remove = original_indices[shuffled_indices_to_remove.cpu()]
    elif args.pruning_method == "random":
        indices_to_remove = np.random.choice(original_indices, math.floor((1-args.data_proportion) * len(original_indices)), replace=False)
    else:
        raise ValueError("Pruning method not recognized")
    
    return indices_to_remove

def train(model : torch.nn.Module,train_dataset,eval_dataset,optimizer,train_attack,eval_attack,args):

 
    model.to(args.device)
    model.train()

    lr_generator = lambda t: np.interp(t, [0, args.num_epochs * args.lr_warmup_end, args.num_epochs], [0, args.lr_max, 0]) #TODO: replace with a proper lr scheduler call

    start_time = time.time()

    # To make sure we keep a correspondence between the indices of the dataset and the indices of the dataloader, we shuffle the dataset before

    train_dataset = PrunableDataset(train_dataset)

    #make eval dataset smaller
    eval_dataset_size = args.eval_size if args.eval_size != -1 else len(eval_dataset)
    eval_dataset =  torch.utils.data.Subset(eval_dataset, np.random.choice(len(eval_dataset), eval_dataset_size, replace=False))
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.early_stopping:
        best_accuracy = 0
        best_epoch = -1
    
    for epoch in range(0, args.num_epochs):

          
        train_dataset_shuffled = ShuffledDataset(train_dataset)

        train_dataloader = torch.utils.data.DataLoader(train_dataset_shuffled, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        train_loader_size = len(train_dataloader)

        is_pruning_epoch = epoch + 1 == args.pruning_epoch and args.data_proportion != 1

        if is_pruning_epoch:
            loss_list = []

        for i, (xs, ys) in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}"):

            lr = lr_generator(epoch + (i+1)/len(train_dataloader))
            optimizer.param_groups[0].update(lr=lr)

            xs, ys = xs.to(args.device), ys.to(args.device)
            
            adv_xs = train_attack.generate_attack(model, xs, ys) 
            logits = model(adv_xs)

            loss = F.cross_entropy(logits, ys, reduction="none")

            if is_pruning_epoch and args.pruning_method != "random":
                loss_list.append(loss)
            
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()


            if args.num_logs_per_epoch > 0 and i > 0 and i % math.ceil((train_loader_size - 1)/args.num_logs_per_epoch) == 0:
                metrics = evaluate(model, eval_dataloader, eval_attack, args)
                metrics["epoch"] = epoch + i/train_loader_size

                if not args.no_wandb:
                    wandb.log(metrics)
                else:
                    print(metrics)
                    
            if i % math.ceil((train_loader_size - 1)/20) == 0:
                if not args.no_wandb:
                    wandb.log({"train_loss": loss, "epoch": epoch + i/train_loader_size, "lr": lr,"dataset_size":len(train_dataset)})
                else:
                    print(f"Epoch {epoch + i/train_loader_size} - loss: {loss} - lr: {lr}")

        end_time = int(time.time()) - int(start_time)

        if not args.no_wandb:
            wandb.log({"epoch": epoch,  "time": end_time})
        else:
            print(f"Epoch {epoch} - time: {end_time}")

        if args.early_stopping:
            
            if metrics["test_accuracy"] > best_accuracy:
                best_accuracy = metrics["test_accuracy"]
                best_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())

            if epoch - best_epoch >= args.early_stopping_patience:
                print("EARLY STOPPING")
                model = best_model
                break
        
        if is_pruning_epoch:
            
            if args.pruning_method != "random":
                loss_tensor = torch.cat(loss_list)
            else:
                loss_tensor = torch.tensor(loss_list)

            shuffled_index = train_dataset_shuffled.get_indices()
            
            if args.systematic_sampling:
                
                original_targets = train_dataset.targets
                shuffled_targets = original_targets[shuffled_index]

                num_classes = np.max(original_targets) + 1

                list_of_indices_to_remove = []
                for i in range(0,num_classes):

                    shuffled_targets_class_mask = shuffled_targets == i
                    original_indices = shuffled_index[shuffled_targets_class_mask]
                    loss_tensor_class = loss_tensor[shuffled_targets_class_mask]

                    indices_to_remove = get_remove_indices(loss_tensor_class,original_indices,args)
                    list_of_indices_to_remove.append(indices_to_remove) 
                
                indices_to_remove = np.concatenate(list_of_indices_to_remove)
            else:
                indices_to_remove = get_remove_indices(loss_tensor,shuffled_index,args)
            
            train_dataset.remove_indices(indices_to_remove)
    
    if args.num_logs_per_epoch == 0:
        final_accuracy = evaluate(model, eval_dataloader, eval_attack, args)["test_accuracy"]
    elif args.early_stopping:
        final_accuracy = best_accuracy
        model = best_model
    else:
        final_accuracy = metrics["test_accuracy"]

    if not args.no_wandb:
        wandb.log({"adv_accuracy": final_accuracy})
    else:
        print(f"Advesraial accuracy: {final_accuracy}")

    return {"model": model, "adv_accuracy": final_accuracy, "class_dist": train_dataset.class_dist()}

            

                








