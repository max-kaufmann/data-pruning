import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from src.data_utils import fetch_sweep_table
import numpy as np
import json

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "axes.unicode_minus": False,
    }
    )
plt.rc('xtick', labelsize=16)    
plt.rc('ytick', labelsize=16)    

#-----MAIN SWEEP PLOT-----

file = r"C:\Users\jamie\OneDrive\Documents\Coding\Python\data-pruning\experiments\wandb_sweeps\logs\xua6wp5m\table.json"
with open (file, "r")  as f:
    all_data = json.loads(f.read())
df = pd.DataFrame.from_dict(all_data)
grouped_data = df.groupby(['Pruning Metric','Data Proportion','Pruning Method'])[['Natural Accuracy','Adversarial Accuracy','Training Time']].agg(['mean', 'std']).reset_index()
print(grouped_data)
for metric in df["Pruning Metric"].unique():
    for dependent_variable in ['Natural Accuracy','Adversarial Accuracy','Training Time']:
        fig, axs = plt.subplots(figsize=(6.8,4.5))
        if metric == 'loss':
            method_loop = ['random','high','low','low+high']
        else:
            method_loop = ['random','low','high','low+high']
        for method in method_loop:
            if metric == 'loss':
                label = (method=='high')*'low' + (method=='low')*'high' + (method=='random')*'random' + (method=='low+high')*'low+high' + " distance"*(method!='random')
            else:
                label = method + " distance"*(method!='random')
            method_data = grouped_data[(grouped_data['Pruning Metric'] == metric) & (grouped_data['Pruning Method'] == method)]
            yerr=method_data[dependent_variable]['std']
            mean=method_data[dependent_variable]['mean']
            axs.plot(method_data['Data Proportion'], mean, label=label, marker='o', linestyle='-')
            axs.plot(method_data['Data Proportion'], mean+yerr, linestyle='-', alpha=0.35, color=plt.gca().lines[-1].get_color())
            axs.plot(method_data['Data Proportion'], mean-yerr, linestyle='-', alpha=0.35, color=plt.gca().lines[-1].get_color())
            axs.fill_between(method_data['Data Proportion'], mean-yerr, mean+yerr, alpha=0.2)

        plt.xlabel('Data Proportion ($p$)',fontsize=20)
        if dependent_variable == 'Training Time':
            dependent_variable = 'Training Time (s)'
        plt.ylabel(dependent_variable,fontsize=20)
        plt.legend(fontsize=15,loc=4)
        plt.grid(True)
        plt.tight_layout()
        axs = plt.gca()
        axs.set_xlim([0.1, 1])
        #axs.set_ylim([ymin, ymax])
        if dependent_variable == 'Natural Accuracy':
            file_name = "natural"
        elif dependent_variable == 'Adversarial Accuracy':
            file_name = "autoattack"
        else:
            file_name = "time"
        plt.savefig(f"plots/cifar10_{metric}_{file_name}.pgf")

#-----PRUNING EPOCH PLOT-----     

# file = r"C:\Users\jamie\OneDrive\Documents\Coding\Python\data-pruning\experiments\wandb_sweeps\logs\1gmx69us\table.json"
# with open (file, "r")  as f:
#     all_data = json.loads(f.read())
# df = pd.DataFrame.from_dict(all_data)
# grouped_data = df.groupby(['Data Proportion','Pruning Epoch'])[['Natural Accuracy','Adversarial Accuracy']].agg(['mean', 'std']).reset_index()
# print(grouped_data)
# fig, axs = plt.subplots(figsize=(6.8,4.5))
    
# for data_prop in [0.3,0.6]:

#     epoch_data = grouped_data[grouped_data['Data Proportion'] == data_prop]
#     yerr=epoch_data['Adversarial Accuracy']['std']
#     mean = epoch_data['Adversarial Accuracy']['mean']
#     axs.plot(epoch_data['Pruning Epoch'], mean, label=f"$p={data_prop}$", marker='o', linestyle='-')
#     axs.fill_between(epoch_data['Pruning Epoch'], mean-yerr, mean+yerr, alpha=0.2)

# plt.xlabel('Pruning Epoch ($e$)',fontsize=20)
# plt.ylabel('AutoAttack Accuracy',fontsize=20)
# plt.tight_layout()
# plt.legend(fontsize=15,loc=4)
# plt.grid(True)
# plt.savefig("plots/cifar10_pruning_epoch.pgf")

#-----DISTANCE LOSS CORRELATION PLOT-----

# fig, axs = plt.subplots(nrows=2,ncols=2)
# for i in range(4):
    
#     file = rf"C:\Users\jamie\OneDrive\Documents\Coding\Python\data-pruning\experiments\wandb_sweeps\logs\distance_loss_correlation\mnist\pe_{i}.json"
#     df = pd.read_json(file)
#     index = [int(bit) for bit in bin(i)[2:].zfill(2)]
#     axs[index[0],index[1]].set_title(f'Epoch {i}', fontstyle='italic')
#     axs[index[0],index[1]].set_xlabel("Distance")
#     axs[index[0],index[1]].set_ylabel("Loss")
#     axs[index[0],index[1]].scatter(1/df["Distance"],df["Loss"],label=f"Pruning Epoch {i}",s=5)

# plt.show()


# file = rf"C:\Users\jamie\OneDrive\Documents\Coding\Python\data-pruning\experiments\wandb_sweeps\logs\distance_loss_correlation\cifar10\pe_{1}.json"
# df = pd.read_json(file)
# fig, axs = plt.subplots()
# axs.set_xlabel("$1/d$",fontsize=20)
# axs.set_ylabel("Loss",fontsize=20)
# axs.scatter(1/df["Distance"],df["Loss"],s=4,rasterized=True)
# axs = plt.gca()
# axs.set_xlim([0, 2])
# axs.set_ylim([0, 5])
# plt.tight_layout()
# plt.savefig("plots/cifar10_distance_loss_correlation.pgf")


