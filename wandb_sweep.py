import argparse
import pandas as pd
import numpy as np
import subprocess as sp
import os
import yaml
import wandb
from src.data_utils import t_test, aggregate_dataframe, mean_class_dist
from src.misc import reset_wandb_env
"""
Script to carry out a wandb sweep as well as logging other metrics which
don't log well to wandb.

"""
def main(args):

    with open(f"./experiments/wandb_sweeps/configs/{args.dataset}_config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)   
    
    sweep_id_full = wandb.sweep(config, project='data-pruning')
    sweep_id = os.path.basename(sweep_id_full)
    os.makedirs(f"./experiments/wandb_sweeps/logs/{sweep_id}/dataframe/")
    wandb.agent(sweep_id_full)
    
    final_run = wandb.init().id
    wandb.finish
    reset_wandb_env()
    
    dataframe = aggregate_dataframe(sweep_id)
    dataframe.to_json(f"./experiments/wandb_sweeps/logs/{sweep_id}/table.json")

    if args.t_test is not None or args.class_dist:
        final_run = wandb.init(id=final_run, resume='allow')
        if args.t_test is not None:
            results = t_test(dataframe,*args.t_test)
            print(results)
            final_run.log({"t Test": wandb.Table(data=results)})
        if args.class_dist:
            class_dist = mean_class_dist(dataframe)
            print(class_dist)
            final_run.log({"Class Distribution":wandb.Table(data=class_dist)})

def parse_args():

    parser = argparse.ArgumentParser(description='Run training script')

    parser.add_argument('--dataset', default='mnist', type=str)
    
    parser.add_argument('--t_test', 
                        nargs=2, 
                        default=None, 
                        help= "Both args should be pruning methods. The second will be tested against the first to see if it gives a larger mean adversarial accuracy.")
    
    parser.add_argument('--class_dist', 
                        action="store_true", 
                        help="Whether to log mean class distribution across repeated runs. class distribution is automatically logged for each individual run."
                        )

    args = parser.parse_args()

    return args
   
if __name__ == "__main__":

    args = parse_args()

    main(args)