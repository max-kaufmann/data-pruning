import matplotlib.pyplot as plt
import pandas as pd
from src.data_utils import fetch_sweep_table
import numpy as np
# loss_table = fetch_sweep_table('jamiestephenson/data-pruning/3s451jg6')
# distance_table = fetch_sweep_table('jamiestephenson/data-pruning/4npv9kyo')
# file = r"C:\Users\jamie\OneDrive\Documents\Coding\Python\data-pruning\experiments\wandb_sweeps\logs\cifar10_distance_p_sweep\table.json"
# df = pd.read_json(file)
# grouped_data = df.groupby(['Data Proportion', 'Pruning Method'])['Adversarial Accuracy'].agg(['mean', 'std']).reset_index()
# fig, ax = plt.subplots() 

# for method in ['random','high','low','low+high']:

#     method_data = grouped_data[grouped_data['Pruning Method'] == method]
#     yerr=method_data['std']
#     ax.plot(method_data['Data Proportion'], method_data['mean'], label=method, marker='o', linestyle='-')
#     ax.fill_between(method_data['Data Proportion'], method_data['mean']-yerr, method_data['mean']+yerr, alpha=0.2)

    
# plt.xlabel('Data Proportion')
# plt.ylabel('Adversarial Accuracy')
# plt.title('Mean Adversarial Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

fig, axs = plt.subplots(nrows=2,ncols=2)
for i in range(4):
    
    file = rf"C:\Users\jamie\OneDrive\Documents\Coding\Python\data-pruning\experiments\wandb_sweeps\logs\distance_loss_correlation\cifar10\pe_{i}.json"
    df = pd.read_json(file)
    index = [int(bit) for bit in bin(i)[2:].zfill(2)]
    axs[index[0],index[1]].set_title(f'Epoch {i}', fontstyle='italic')
    axs[index[0],index[1]].set_xlabel("Distance")
    axs[index[0],index[1]].set_ylabel("Loss")
    axs[index[0],index[1]].scatter(df["Distance"],df["Loss"],label=f"Pruning Epoch {i}",s=5)

plt.show()

