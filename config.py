import os

logging_categories = "name,attack,avg_loss,accuracy"
device = "cpu" # Set as global variable during initalisation.

project_path = os.path.dirname(os.path.realpath(__file__))
# Adding the trailing slash to make it compatible with other usages
project_path = os.path.join(project_path, '') 
#Get a list of the current attacks in the repository
file_list = os.listdir(os.path.join(project_path, "./attacks"))
files_to_remove = ["__init__.py","attacks.py"]

file_list = [file for file in file_list if file.endswith('.py') and file not in files_to_remove]

attack_list = [ file[:-3] for file in file_list]
attack_list = ["none"] + attack_list

pruning_methods = ["low","high","low+high","random"]
project_path = os.path.dirname(os.path.realpath(__file__))