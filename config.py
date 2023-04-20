import os

logging_categories = "name,attack,avg_loss,accuracy"
min = 0
max = 1
device = "cpu" # Set as global variable during initalisation.

project_path = os.path.dirname(os.path.realpath(__file__))
# Adding the trailing slash to make it compatible with other usages
project_path = os.path.join(project_path, '') 
#Get a list of the current attacks in the repository
file_list = os.listdir(os.path.join(project_path, "./attacks"))
file_list.remove("__init__.py")
file_list.remove("__pycache__")
file_list.remove("attacks.py")
file_list.remove("README.md")

attack_list = [ file[:-3] for file in file_list]
attack_list = ["none"] + attack_list
