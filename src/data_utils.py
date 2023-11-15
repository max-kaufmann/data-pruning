
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
        num_classes = np.max(self.data.targets) + 1
        class_count = torch.tensor([None]*num_classes)
        for i in range(0,num_classes):
            class_count[i] = sum(self.data.targets == i)
        return class_count/sum(class_count)

def download_table(run, current_analysis):
    try:
        run_name = f"run-{run.id}"
        run_path = Path(run.dir).parts
        table_remote = current_analysis.use_artifact(
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

def download_sweep_table(sweep, current_analysis):
    runs = sweep.runs
    tables = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(download_table, run, current_analysis) for run in runs]
        for future in concurrent.futures.as_completed(futures):
            table_local = future.result()
            if table_local is not None:
                tables.append(table_local)
    sweep_table = pd.concat(tables).reset_index(drop=True)
    sweep_table.to_csv("table.csv")
    shutil.rmtree('./artifacts')

     