import argparse
import datetime
import logging
import os
import yaml
import wandb
import numpy as np
from lib.utils import set_logging


def load_config(config='config.yaml'):
    with open(config, 'r') as file:
        return yaml.safe_load(file)


def setup_directories(config, seed):
    levels = config["data"]["levels"]
    ablation_suffix = "-ablation" if config["data"]["ablation"] else ""
    config["data"]["random_seed"] = seed
    save_dir = f"result/{config['data']['save_dir']}-l{levels}{ablation_suffix}-seed{seed}"
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
    save_dir = os.path.join(save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def initialize_logging(save_dir, config):
    set_logging(save_dir)
    logging.info(f"Save directory: {save_dir}")

    if config["wandb_log"]:
        wandb.init(project="multi-fidelity", config=config)
        logging.info(f"Running wandb experiment:\n{wandb.run.get_url()}")


def get_supervisor(config, save_dir):
    if config["data"]["ablation"]:
        from model.AblationSupervisor import Supervisor
    else:
        from model.supervisor import Supervisor

    return Supervisor(save_dir, **config)


def perform_training(supervisor):
    try:
        supervisor.train()
    except KeyboardInterrupt:
        supervisor.save_model()

def calculate_statistics(test_nrmses, config, levels, ablation):
    tests = np.array(test_nrmses)
    result_path = f"./result/test/{config['data']['save_dir']}-l{levels}{ablation}"
    os.makedirs(result_path, exist_ok=True)
    set_logging(result_path)

    mean, std = np.mean(tests), np.std(tests)
    return mean, std

def get_training_time(start_time):
    training_duration = datetime.datetime.now() - start_time
    total_seconds = int(training_duration.total_seconds())

    # Extracting minutes and seconds
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    # Formatting as "minutes:seconds"
    formatted_time = f"{minutes}:{seconds:02d}"
    return formatted_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--seed", type=int)
    parser.add_argument("--levels", type=int)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()
    config = load_config(args.config)
    
    # config["data"]["random_seed"] = args.seed
    config["data"]["data_path"] = args.data_path
    config["data"]["save_dir"] = args.save_dir
    config["data"]["levels"] = args.levels
    config["model"]["device"] = args.device

    test_nrmses = []
    total_times = []

    for seed in [1, 2, 3]:
        save_dir = setup_directories(config, seed)
        initialize_logging(save_dir, config)

        supervisor = get_supervisor(config, save_dir)
        start_time = datetime.datetime.now()

        perform_training(supervisor)

        formatted_time = get_training_time(start_time)
        total_times.append(formatted_time)
        logging.info(f"Training time: {formatted_time}")

        test_nrmse = supervisor.best_loss_dict[f"l{supervisor.levels}_nrmse_loss"]
        test_nrmses.append(test_nrmse)
        logging.info(f"Test NRMSE: {test_nrmse}")

        if config["wandb_log"]:
            wandb.finish()

    levels = config["data"]["levels"]
    ablation = "-ablation" if config["data"]["ablation"] else ""
    mean, std = calculate_statistics(test_nrmses, config, levels, ablation)
    logging.info(f"Training times: {total_times}")
    logging.info(f"Test NRMSE: {test_nrmses}")
    logging.info(f"Mean: {mean}, Std: {std}")
