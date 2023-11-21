
import numpy as np 
import os
import torch
import pandas as pd
from pathlib import Path
import json
import wandb
from tqdm import tqdm
import concurrent.futures
import shutil
from scipy.stats import ttest_ind

class ShuffledDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = torch.randperm(len(self.dataset))

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.dataset)

    def get_indices(self):
        return self.indices

class PrunableDataset(torch.utils.data.Dataset):
    
    def __init__(self,dataset):
        self.data = dataset
        self.targets = np.array(self.data.targets)

    def __getitem__(self,index):
        data,target = self.data[index]
        return data,target

    def __len__(self):
        return len(self.data)

    def remove_indices(self,indices):
        mask = np.ones(shape=len(self.data),dtype=bool)
        mask[indices] = False
        self.data.data = self.data.data[mask]
        self.data.targets = self.targets[mask]

    def class_dist(self):
        classes = np.unique(self.data.targets)
        return [sum(self.data.targets == i)/self.__len__() for i in classes]


#the next two functions are adapted from: 
#https://community.wandb.ai/t/custom-analysis-over-sweep/5215/9

def download_table(run, current_init):
    try:
        run_name = f"run-{run.id}"
        run_path = Path(run.dir).parts
        table_remote = current_init.use_artifact(
            f"{run_path[-3]}/{run_path[-2]}/{run_name}-Table:v0", type="run_table"
        )
        table_dir = table_remote.download()
        table_json = json.load(
            open(f"{table_dir}/Table.table.json")
        )
        table_wandb = wandb.Table.from_json(table_json, table_remote)
        table_local = pd.DataFrame(
            data=table_wandb.data, columns=table_wandb.columns
        ).copy()
        return table_local
    except Exception as e:
        print(f"Error downloading table for run {run.id}: {e}")
        return None

def fetch_sweep_table(sweep_id):
    """A janky way to fetch the data from a wandb sweep table as a pandas dataframe.
    sweep_id must be of the form: 'entity/project/sweep_id' ."""
    api = wandb.Api()
    sweep: wandb.sweep = api.sweep(sweep_id)
    current_init = wandb.init(name=f"Fetching table from {sweep_id}")
    runs = sweep.runs
    tables = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(download_table, run, current_init) for run in runs]
        for future in concurrent.futures.as_completed(futures):
            table_local = future.result()
            if table_local is not None:
                tables.append(table_local)
    sweep_table = pd.concat(tables).reset_index(drop=True)
    shutil.rmtree('./artifacts')
    return sweep_table

def aggregate_dataframe(sweep_name):
    """The way train.py is set up means that a wandb sweep gives a json file of data for
    each run. This function aggregates them into one pandas dataframe and then deletes them."""
    path = f"./experiments/wandb_sweeps/logs/{sweep_name}/dataframe/"
    frames = [pd.read_json(path+f) for f in os.listdir(path)]
    shutil.rmtree(path)
    return pd.concat(frames).reset_index(drop=True)

def t_test(dataframe, pruning_method_1, pruning_method_2):
    """
    This is a function to carry out a one sided Welch's t test on the data in a pandas dataframe.
    H_0: pruning method 1 and 2 give adversarial accuracies with the same mean.
    H_1: pruning method 2 gives adversarial accuracies with higher mean.

    """
    test_results = pd.DataFrame(columns=["Data Proportion", "t Value", "p Value"])

    for dp in sorted(set(dataframe["Data Proportion"])):
        sample1 = dataframe[(dataframe["Pruning Method"]==pruning_method_1) & (dataframe["Data Proportion"]==dp)]
        sample2 = dataframe[(dataframe["Pruning Method"]==pruning_method_2) & (dataframe["Data Proportion"]==dp)]
        test_results.loc[len(test_results)] = [dp,*list(ttest_ind(sample1['Adversarial Accuracy'], sample2['Adversarial Accuracy'], equal_var=False, alternative='less')) ]
    return test_results

def mean_class_dist(dataframe):
    mean_class_dist = pd.DataFrame(columns=["Pruning Method", "Data Proportion", "Mean Class Distribution"])
    for dp in sorted(set(dataframe["Data Proportion"])):
        for pm in set(dataframe["Pruning Method"]):
            class_dists = dataframe[(dataframe["Pruning Method"]==pm) & (dataframe["Data Proportion"]==dp)]
            arr = np.array(class_dists['Class Distribution'])
            mean_class_dist.loc[len(mean_class_dist)] = [pm,dp,np.mean(arr, axis=0).tolist()]
    return mean_class_dist


     