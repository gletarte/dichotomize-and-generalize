import random
import os
from os.path import join, abspath, dirname, exists

import click
from sklearn.model_selection import ParameterGrid

from experiment import launch
from pbgdeep.utils import get_logging_dir_name

RESULTS_PATH = os.environ.get('PBGDEEP_RESULTS_DIR', join(dirname(abspath(__file__)), "results"))

@click.command()
@click.option('--dataset', type=str, default=None, help="Filter to launch for a particular dataset.")
@click.option('--network', type=str, default=None, help="Filter to launch for a particular network.")
@click.option('--gpu-device', type=int, default=0, help="GPU device id to run on.")
@click.pass_context
def main(ctx, dataset, network, gpu_device):
    random.seed(42)
    experiment_name = "neurips"
    datasets = ['ads', 'mnist17', 'mnist49', 'mnist56', 'adult', 'mnist_low_high']
    hidden_size = [10, 50, 100]
    hidden_layers = [1, 2, 3]
    sample_size = [10, 50, 100, 1000, 10000]
    epochs = 150
    batch_size = [32]
    weight_decay = [0, 1e-4, 1e-6]
    learning_rate = [0.1, 0.01, 0.001]
    lr_patience = 5
    stop_early = 20
    optim_algo = ['sgd', 'adam']
    valid_size = [0.2]
    pre_epochs = [20]
    random_seed = [42]


    param_grid = ParameterGrid([{'network': ['pbgnet'], 'dataset': datasets, 'hidden_size': hidden_size,
                                 'sample_size': sample_size, 'batch_size': batch_size, 'weight_decay': [0.0],
                                 'prior': ['init'], 'learning_rate': learning_rate, 'optim_algo': ['adam'],
                                 'valid_size': valid_size, 'pre_epochs': [0], 'hidden_layers': hidden_layers,
                                 'random_seed': random_seed},
                                {'network': ['pbgnet'], 'dataset': datasets, 'hidden_size': hidden_size,
                                 'sample_size': sample_size, 'batch_size': batch_size, 'weight_decay': [0.0],
                                 'prior': ['pretrain'], 'learning_rate': learning_rate, 'optim_algo': ['adam'],
                                 'valid_size': [0.5], 'pre_epochs': pre_epochs, 'hidden_layers': hidden_layers,
                                 'random_seed': random_seed},
                                {'network': ['pbgnet_ll'], 'dataset': datasets, 'hidden_size': hidden_size,
                                 'sample_size': sample_size, 'batch_size': batch_size, 'weight_decay': weight_decay,
                                 'prior': ['init'], 'learning_rate': learning_rate, 'optim_algo': ['adam'],
                                 'valid_size': valid_size, 'pre_epochs': [0], 'hidden_layers': hidden_layers,
                                 'random_seed': random_seed},
                                {'network': ['baseline'], 'dataset': datasets, 'hidden_size': hidden_size,
                                 'sample_size': [0], 'batch_size': batch_size, 'weight_decay': weight_decay,
                                 'prior': ['zero'], 'learning_rate': learning_rate, 'optim_algo': optim_algo,
                                 'valid_size': valid_size, 'pre_epochs': [0], 'hidden_layers': hidden_layers,
                                 'random_seed': random_seed}])


    param_grid = [t for t in param_grid if (dataset is None or dataset == t['dataset'])
                                        and (network is None or network == t['network'])]
    ordering = {d: i for i, d in enumerate(datasets)}
    param_grid = sorted(param_grid, key=lambda d: ordering[d['dataset']])

    n_tasks = len(param_grid)
    for i, task_dict in enumerate(param_grid):
        kwargs = dict(dataset=task_dict['dataset'],
                      experiment_name=experiment_name,
                      network=task_dict['network'],
                      hidden_size=task_dict['hidden_size'],
                      hidden_layers=task_dict['hidden_layers'],
                      sample_size=task_dict['sample_size'],
                      weight_decay=task_dict['weight_decay'],
                      prior=task_dict['prior'],
                      learning_rate=task_dict['learning_rate'],
                      lr_patience=lr_patience,
                      optim_algo=task_dict['optim_algo'],
                      epochs=epochs,
                      batch_size=(64 if task_dict['dataset'] in ['adult', 'mnist_low_high'] else 32),
                      valid_size=task_dict['valid_size'],
                      pre_epochs=task_dict['pre_epochs'],
                      stop_early=stop_early,
                      gpu_device=gpu_device,
                      random_seed=task_dict['random_seed'],
                      logging=True)
        print(f"Launching task {i + 1}/{n_tasks}: {kwargs}")

        directory_name = get_logging_dir_name(kwargs)
        logging_path = join(RESULTS_PATH, experiment_name, kwargs['dataset'], directory_name)

        if exists(join(logging_path, "done.txt")):
            print("Task already computed, skipping...")
        else:
            ctx.invoke(launch, **kwargs)

    print("### ALL TASKS DONE ###")

if __name__ == '__main__':
    main()
