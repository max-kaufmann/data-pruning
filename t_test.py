import os
import wandb
import pandas as pd
import argparse
from scipy.stats import ttest_ind
from src.data_utils import download_sweep_table


"""
This is a script to carry out a one sided Welch's t test on the data in a wandb.Table object
associated with a wandb sweep that has already finished.

"""

parser = argparse.ArgumentParser()

parser.add_argument("--sweep_id", type=str)
parser.add_argument("--entity", default="jamiestephenson", type=str, help="wandb user's account name")
parser.add_argument("--project", default="data-pruning", type=str, help="wandb project name")
parser.add_argument("--pruning_method_1", 
                    default="random", 
                    type=str, 
                    help="The pruning method for which the mean is lower in the alternative hypothesis."
                    )
parser.add_argument("--pruning_method_2", 
                    default="high", 
                    type=str, 
                    help="The pruning method for which the mean is higher in the alternative hypothesis."
                    )

args = parser.parse_args()

sweep_id = f"{args.entity}/{args.project}/sweeps/{args.sweep_id}"

wandb.login()
api = wandb.Api()
sweep: wandb.sweep = api.sweep(sweep_id)
current_analysis = wandb.init(name=f"t Test for Sweep {args.sweep_id}")

download_sweep_table(sweep, current_analysis)
table = pd.read_csv("table.csv")

test_results = pd.DataFrame(columns=["Data Proportion", "t Value", "p Value"])

for dp in sorted(set(table["Data Proportion"])):
    sample1 = table[(table["Pruning Method"]==args.pruning_method_1) & (table["Data Proportion"]==dp)]
    sample2 = table[(table["Pruning Method"]==args.pruning_method_2) & (table["Data Proportion"]==dp)]
    test_results.loc[len(test_results)] = [dp,*list(ttest_ind(sample1['Adversarial Accuracy'], sample2['Adversarial Accuracy'], equal_var=False, alternative='less')) ]

os.remove("table.csv")
current_analysis.log({"t Test": wandb.Table(data=test_results)})

