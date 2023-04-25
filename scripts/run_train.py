import argparse
import jsonlines
import os 
from sweep import project_dir


def parse_args():

    parser = argparse.ArgumentParser(description='Run training script')

    parser.add_argument('--file', type=str, required=True, help='Path to sweep file')

    parser.add_argument('--job_id', type=int, required=True, help='Job ID')

    parser.add_argument('--task_id', type=int, required=True, help='Task ID')

    args = parser.parse_args()

    return args

def main(config):

    base_command = f"python " + str(os.path.join(project_dir,"train.py"))

    command_args = []
    for key, value in config.items():

        if isinstance(value, bool):
            if value:
                command_args.append(f"\'--{key}\'")
        else:
            command_args.append(f"\'--{key}\'")
            command_args.append(f"\'{value}\'")

    command = base_command + " " + " ".join(command_args)

    print(command)
    os.system(command)


if __name__ == "__main__":

    args = parse_args()
    
    with jsonlines.open(args.file) as reader:
        f = list(reader)
        config = f[args.task_id]
    
    
    config["experiment_name"] = config["experiment_name"] + f" {args.job_id}_{args.task_id}"

    main(config)



