import matplotlib.pyplot as plt
import pandas as pd
from src.data_utils import fetch_sweep_table
loss_table = fetch_sweep_table('jamiestephenson/data-pruning/3s451jg6')
distance_table = fetch_sweep_table('jamiestephenson/data-pruning/4npv9kyo')
#file = r"C:\Users\jamie\OneDrive\Documents\Coding\Python\data-pruning\experiments\wandb_sweeps\logs\92b40oh7\table.json"
#df = pd.read_json(file)
loss_grouped_data = loss_table.groupby(['Data Proportion', 'Pruning Method'])['Adversarial Accuracy'].agg(['mean', 'std']).reset_index()
distance_grouped_data = distance_table.groupby(['Data Proportion', 'Pruning Method'])['Adversarial Accuracy'].agg(['mean', 'std']).reset_index()
df = pd.concat([loss_grouped_data,distance_grouped_data]).reset_index(drop=True)
fig, ax = plt.subplots() 

for method in ['random','high','low']:

    method_data = df[df['Pruning Method'] == method]
    yerr=method_data['std']
    ax.plot(method_data['Data Proportion'], method_data['mean'], label=method, marker='o', linestyle='-')
    ax.fill_between(method_data['Data Proportion'], method_data['mean']-yerr, method_data['mean']+yerr, alpha=0.2)
    

plt.xlabel('Data Proportion')
plt.ylabel('Mean Adversarial Accuracy')
plt.title('Mean Adversarial Accuracy')
plt.legend()
plt.grid(True)
plt.show()