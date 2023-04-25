import subprocess
import yaml
from itertools import product
import json
import argparse
import os
# import time
# import base64
from datetime import datetime
import jsonlines
import debugpy
from pathlib import Path

project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def attach_debugger(port=5678):
    debugpy.listen(port)

    print(f"Waiting for debugger on port {port}")

    debugpy.wait_for_client()
    print(f"Debugger attached on port {port}")

def sweep(config_yaml: str, args):

    with open(config_yaml) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_dir = os.path.dirname(config_yaml)
    param_combinations = product(*config['hyperparameters'].values())
    sweeps = [dict(zip(config['hyperparameters'].keys(), values)) for values in param_combinations]

    for sweep in sweeps:
        sweep["experiment_name"] = args.experiment_name

    # Check that all data files exist, this has errored me out enough times that I think it's worth an assert


    sweep_file_dir = os.path.join(config_dir, 'sweep_configs')
    if not os.path.exists(sweep_file_dir):
        os.makedirs(sweep_file_dir)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_file = os.path.join(sweep_file_dir, f'{current_time}.json')

    if os.path.isfile(sweep_file):
        os.remove(sweep_file)

    i = 0
    while os.path.isfile(sweep_file):
        i += 1
        sweep_file = os.path.join(sweep_file_dir, f'{current_time}_{i}.json')


    with jsonlines.open(sweep_file, 'w') as writer:
        for sweep in sweeps:
            writer.write(sweep)
    
    run_directory = project_dir / 'scripts'

    partition = 'compute' if not args.run_interactive else 'interactive'

    slurm_script = run_directory / 'agent.sh'

    log_dir = os.path.join(os.path.dirname(os.path.dirname(sweep_file)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    time_limit = (
        f"0-{args.time_limit}:00:00" if not args.run_interactive else "0-00:30:00"
    )
    if args.node_list is None:
        command = [
            'sbatch',
            f'--gpus={config["fixed_parameters"]["num_gpus"]}',
            '--array',
            f'0-{len(sweeps) - 1}',
            f'--cpus-per-gpu',
            f'{config["fixed_parameters"]["cpus_per_gpu"]}',
            f'--mem={config["fixed_parameters"]["ram_limit_gb"]}G',
            '--partition',
            partition,
            '--output',
            os.path.join(log_dir, '%A_%a.log'),
            '--time',
            time_limit,
            slurm_script,
            config['project_name'],
            sweep_file,
            os.environ['WANDB_API_KEY'],
        ]

        print(command)
        subprocess.run(command)
    else:
        job_num = 0
        while job_num < len(sweeps):
            command = ['sbatch',
                        '--nodes=1',
                        f'--gpus={config["fixed_parameters"]["num_gpus"]}',
                        '--array',
                        f'{job_num}-{job_num}',
                        '--cpus-per-gpu',
                        f'{config["fixed_parameters"]["cpus_per_gpu"]}',
                        f'--mem={config["fixed_parameters"]["ram_limit_gb"]}G',
                        f'-w',
                        f'compute-permanent-node-{args.node_list[job_num % len(args.node_list)]}',
                        '--partition',
                        partition,
                        '--output',
                        os.path.join(log_dir, '%A_%a.log'),
                        '--time',
                        time_limit,

                        slurm_script,
                        config['project_name'],
                        sweep_file,
                        os.environ['WANDB_API_KEY'],
                        ]
            print(command)
            job_num += 1

            subprocess.run(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default='experiments/sweeps')
    parser.add_argument("--experiment_type", type=str, required=False, default='mnist')
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=False, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument("--run_interactive", action="store_true", default=False)
    parser.add_argument("--node_list", type=str, required=False, default=None)
    parser.add_argument("--time_limit", type=int, required=False, default=24)

    args = parser.parse_args()

    if args.debug:
        attach_debugger(port=args.debug_port)

    args.node_list = args.node_list.split(",") if args.node_list is not None else None
    args.experiment_dir = os.path.join(project_dir, args.experiment_dir)

    for config_file in os.listdir(os.path.join(args.experiment_dir, args.experiment_type)):
        if config_file.endswith(".yaml"):
            if args.config_name is None or config_file == args.config_name + ".yaml":
                experiment_file = os.path.join(args.experiment_dir, args.experiment_type, config_file)
                sweep(experiment_file, args)