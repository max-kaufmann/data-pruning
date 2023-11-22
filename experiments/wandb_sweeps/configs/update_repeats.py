import yaml
#script to update number of repeats in wandb sweep config file
n = 120  
config_file = '.\experiments\wandb_sweeps\configs\mnist_config.yaml'
with open(config_file, 'r') as file:
    existing_data = yaml.safe_load(file)
numbers = list(range(1, n+1))
existing_data['parameters']['repeat']['values'] = numbers
with open(config_file, 'w') as file:
    yaml.dump(existing_data, file)

    