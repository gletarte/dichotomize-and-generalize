import json

import os
from os.path import join, abspath, dirname, exists
from os import makedirs

import click
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

from poutyne.framework import Model, Experiment
from poutyne.framework.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from poutyne.layers import Lambda

from pbgdeep.dataset_loader import DatasetLoader
from pbgdeep.networks import PBGNet, BaselineNet, PBCombiNet
from pbgdeep.utils import linear_loss, accuracy, get_logging_dir_name, MasterMetricLogger, MetricLogger

RESULTS_PATH = os.environ.get('PBGDEEP_RESULTS_DIR', join(dirname(abspath(__file__)), "results"))

@click.command()
@click.option('-d', '--dataset', type=str, default="breast", help="Name of the dataset to use.")
@click.option('--experiment-name', type=str, default="test", help="Name of the experiment (for logging).")
@click.option('-n', '--network', type=click.Choice(['pbgnet', 'pbgnet_ll', 'baseline', 'pbcombinet', 'pbcombinet_ll']),
              default='pbgnet', help="Name of the network architecture to use.")
@click.option('--hidden-size', type=int, default=10, help="Size of the hidden layers (number of neurons).")
@click.option('--hidden-layers', type=int, default=1, help="Number of hidden layers (depth of the network).")
@click.option('--sample-size', type=int, default=100, help="Sample size T for stochastic approximation of PBGNet.")
@click.option('--weight-decay', type=float, default=0, help="Weight decay (L2 penalty).")
@click.option('--prior', type=click.Choice(['zero', 'init', 'pretrain']), default='init', help="Prior distribution P.")
@click.option('--learning-rate', type=float, default=0.01, help="Learning rate.")
@click.option('--lr-patience', type=int, default=20, help="Learning rate scheduler patience before halving.")
@click.option('--optim-algo', type=click.Choice(['sgd', 'adam']), default='adam', help="Optimization algorithm.")
@click.option('--epochs', type=int, default=10, help="Maximum number of epochs.")
@click.option('--batch-size', type=int, default=8, help="Batch size.")
@click.option('--valid-size', type=float, default=0.2, help="Validation set size (pretrain set size when pretraining).")
@click.option('--pre-epochs', type=int, default=5, help="Pretrain number of epochs.")
@click.option('--stop-early', type=int, default=0, help="Early stopping patience.")
@click.option('--gpu-device', type=int, default=0, help="GPU device id to run on.")
@click.option('--random-seed', type=int, default=42, help="Random seed for reproducibility.")
@click.option('--logging', type=bool, default=True, help="Logging flag.")
def launch(dataset, experiment_name, network, hidden_size, hidden_layers, sample_size, weight_decay, prior,\
           learning_rate, lr_patience, optim_algo, epochs, batch_size, valid_size, pre_epochs, stop_early,\
           gpu_device, random_seed, logging):

    # Setting random seed for reproducibility
    random_state = check_random_state(random_seed)
    torch.manual_seed(random_seed)

    # Pac-Bayes Bound parameters
    delta = 0.05
    C_range = torch.Tensor(np.arange(0.1, 20.0, 0.01))

    # Setting GPU device
    device = None
    if torch.cuda.is_available() and gpu_device != -1:
        torch.cuda.set_device(gpu_device)
        device = torch.device('cuda:%d' % gpu_device)
        print("Running on GPU %d" % gpu_device)
    else:
        print("Running on CPU")

    # Logging
    experiment_setting = dict([('experiment_name', experiment_name), ('dataset', dataset), ('network', network),
                               ('hidden_size', hidden_size), ('hidden_layers', hidden_layers),
                               ('sample_size', sample_size), ('epochs', epochs), ('weight_decay', weight_decay),
                               ('prior', prior), ('learning_rate', learning_rate), ('lr_patience', lr_patience),
                               ('optim_algo', optim_algo), ('batch_size', batch_size), ('valid_size', valid_size),
                               ('pre_epochs', pre_epochs), ('stop_early', stop_early), ('random_seed', random_seed)])

    directory_name = get_logging_dir_name(experiment_setting)

    logging_path = join(RESULTS_PATH, experiment_name, dataset, directory_name)
    if logging:
        if not exists(logging_path): makedirs(logging_path)
        with open(join(logging_path, "setting.json"), 'w') as out_file:
            json.dump(experiment_setting, out_file, sort_keys=True, indent=4)

    # Loading dataset
    dataset_loader = DatasetLoader(random_state=random_state)
    X_train, X_test, y_train, y_test = dataset_loader.load(dataset)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                          y_train,
                                                          test_size=valid_size,
                                                          random_state=random_state)

    # Experiment
    batch_metrics = [accuracy]
    epoch_metrics = []
    save_every_epoch = False
    cost_function = linear_loss
    monitor_metric = 'val_loss'
    valid_set_use = 'val'
    callbacks = []

    if network in ['pbgnet', 'pbcombinet']:
        print("### Using Pac-Bayes Binary Gradient Network ###")
        if prior in ['zero', 'init']:
            valid_set_use = 'train'
            X_train = np.vstack([X_train, X_valid])
            y_train = np.vstack([y_train, y_valid])
        elif prior == 'pretrain':
            valid_set_use = 'pretrain'

        if network == 'pbgnet':
            net = PBGNet(X_train.shape[1], hidden_layers * [hidden_size], X_train.shape[0], sample_size, delta)
        else:
            net = PBCombiNet(X_train.shape[1], hidden_layers * [hidden_size], X_train.shape[0], delta)
        monitor_metric = 'bound'
        cost_function = net.bound
        epoch_metrics.append(MasterMetricLogger(network=net,
                                                loss_function=linear_loss,
                                                delta=delta,
                                                n_examples=X_train.shape[0]))

    elif network in ['pbgnet_ll', 'pbcombinet_ll']:
        print("### Using PAC-Bayes Gradient Network Architecture and Optimizing Linear Loss ###")
        if network == 'pbgnet_ll':
            net = PBGNet(X_train.shape[1], hidden_layers * [hidden_size], X_train.shape[0], sample_size, delta)
        else:
            net = PBCombiNet(X_train.shape[1], hidden_layers * [hidden_size], X_train.shape[0], delta)
        epoch_metrics.append(MasterMetricLogger(network=net,
                                                loss_function=linear_loss,
                                                delta=delta,
                                                n_examples=X_train.shape[0],
                                                C_range=C_range))
        callbacks.append(ModelCheckpoint(join(logging_path, 'bound_checkpoint_epoch.ckpt'),
                                         temporary_filename=join(logging_path, 'bound_checkpoint_epoch.tmp.ckpt'),
                                         monitor='bound',
                                         mode='min',
                                         save_best_only=True))
    elif network == "baseline":
        print("### Running the Baseline Network with Tanh activations ###")
        net = BaselineNet(X_train.shape[1], hidden_layers * [hidden_size], torch.nn.Tanh)

    if network.startswith('pb'):
        epoch_metrics.append(MetricLogger(network=net, key='bound'))
        epoch_metrics.append(MetricLogger(network=net, key='kl'))
        epoch_metrics.append(MetricLogger(network=net, key='C'))

    # Parameters initialization
    if prior in ['zero', 'init']:
        net.init_weights()

    elif prior == 'pretrain':
        print("### Pre-training network ###")
        if network == 'pbgnet':
            pre_net = PBGNet(X_valid.shape[1], hidden_layers * [hidden_size], X_valid.shape[0], sample_size, delta)
        else:
            pre_net = PBCombiNet(X_valid.shape[1], hidden_layers * [hidden_size], X_valid.shape[0], delta)

        pre_net.init_weights()
        pre_optimizer = torch.optim.Adam(pre_net.parameters(), lr=learning_rate, weight_decay=0.0)
        pre_logging_path = join(logging_path, 'pretrain')
        if not exists(pre_logging_path): makedirs(pre_logging_path)

        pretrain = Experiment(directory=pre_logging_path,
                              module=pre_net,
                              optimizer=pre_optimizer,
                              loss_function=linear_loss,
                              monitor_metric='loss',
                              device=device,
                              logging=logging,
                              batch_metrics=[accuracy])

        pretrain_loader = DataLoader(TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid)),
                                     batch_size,
                                     shuffle=True)

        pretrain.train(train_loader=pretrain_loader,
                       valid_loader=None,
                       epochs=pre_epochs,
                       save_every_epoch=False,
                       disable_tensorboard=True,
                       seed=random_seed)

        history = pd.read_csv(pretrain.log_filename, sep='\t')
        best_epoch_index = history['loss'].idxmin()
        best_epoch_stats = history.iloc[best_epoch_index:best_epoch_index + 1]
        best_epoch = best_epoch_stats['epoch'].item()
        ckpt_filename = pretrain.best_checkpoint_filename.format(epoch=best_epoch)
        weights = torch.load(ckpt_filename, map_location='cpu')

        net.load_state_dict(weights, strict=False)

    print("### Training ###")

    # Setting prior
    if network.startswith('pb') and prior in ['init', 'pretrain']:
        net.set_priors(net.state_dict())

    # Adding early stopping and lr scheduler
    reduce_lr = ReduceLROnPlateau(monitor=monitor_metric, mode='min', patience=lr_patience, factor=0.5, \
                                  threshold_mode='abs', threshold=1e-4, verbose=True)
    lr_schedulers = [reduce_lr]

    early_stopping = EarlyStopping(monitor=monitor_metric,
                                   mode='min',
                                   min_delta=1e-4,
                                   patience=stop_early,
                                   verbose=True)
    if stop_early > 0:
        callbacks.append(early_stopping)

    # Initializing optimizer
    if optim_algo == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optim_algo == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Creating Poutyne experiment
    expt = Experiment(directory=logging_path,
                      module=net,
                      optimizer=optimizer,
                      loss_function=cost_function,
                      monitor_metric=monitor_metric,
                      device=device,
                      logging=logging,
                      batch_metrics=batch_metrics,
                      epoch_metrics=epoch_metrics)

    # Initializing data loaders
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)), batch_size, shuffle=True)
    valid_loader = None
    if valid_set_use == 'val':
        valid_loader = DataLoader(TensorDataset(torch.Tensor(X_valid), torch.Tensor(y_valid)), batch_size)

    # Launching training
    expt.train(train_generator=train_loader,
               valid_generator=valid_loader,
               epochs=epochs,
               callbacks=callbacks,
               lr_schedulers=lr_schedulers,
               save_every_epoch=save_every_epoch,
               disable_tensorboard=True,
               seed=random_seed)

    print("### Testing ###")
    sign_act_fct = lambda: Lambda(lambda x: torch.sign(x))
    test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test)), batch_size)

    if network == 'baseline':
        expt.test(test_generator=test_loader, checkpoint='best', seed=random_seed)
        # Binary network testing (sign activation)
        best_epoch = expt.get_best_epoch_stats()['epoch'].item()
        ckpt_filename = expt.best_checkpoint_filename.format(epoch=best_epoch)
        binary_net = BaselineNet(X_test.shape[1], hidden_layers * [hidden_size], sign_act_fct)
        weights = torch.load(ckpt_filename, map_location='cpu')
        binary_net.load_state_dict(weights, strict=False)
        binary_model = Model(binary_net, 'sgd', linear_loss, batch_metrics=[accuracy])
        test_loss, test_accuracy = binary_model.evaluate_generator(test_loader, steps=None)
        test_stats = pd.read_csv(expt.test_log_filename, sep='\t')
        test_stats['bin_test_linear_loss'] = test_loss
        test_stats['bin_test_accuracy'] = test_accuracy
        test_stats['linear_loss'] = test_stats['loss']
        test_stats['val_linear_loss'] = test_stats['val_loss']
        test_stats['test_linear_loss'] = test_stats['test_loss']
        test_stats.to_csv(expt.test_log_filename, sep='\t', index=False)

    def pbgnet_testing(target_metric, irrelevant_columns, n_repetitions=20):
        print(f"Restoring best model according to {target_metric}")

        # Cleaning logs
        history = pd.read_csv(expt.log_filename, sep='\t').drop(irrelevant_columns, axis=1, errors='ignore')
        history.to_csv(expt.log_filename, sep='\t', index=False)

        # Loading best weights
        best_epoch_index = history[target_metric].idxmin()
        best_epoch_stats = history.iloc[best_epoch_index:best_epoch_index + 1].reset_index(drop=True)
        best_epoch = best_epoch_stats['epoch'].item()
        print(f"Found best checkpoint at epoch: {best_epoch}")
        ckpt_filename = expt.best_checkpoint_filename.format(epoch=best_epoch)
        if network in ['pbgnet_ll', 'pbcombinet_ll'] and target_metric == 'bound':
            ckpt_filename = join(logging_path, 'bound_checkpoint_epoch.ckpt')
        weights = torch.load(ckpt_filename, map_location='cpu')

        # Binary network testing (sign activation)
        binary_net = BaselineNet(X_test.shape[1], hidden_layers * [hidden_size], sign_act_fct)
        updated_weights = {}
        for name, weight in weights.items():
            if name.startswith('layers'):
                name = name.split('.', 2)
                name[1] = str(2 * int(name[1]))
                name = '.'.join(name)
                updated_weights[name] = weight

        binary_net.load_state_dict(updated_weights, strict=False)
        binary_model = Model(binary_net, 'sgd', linear_loss, batch_metrics=[accuracy])
        test_loss, test_accuracy = binary_model.evaluate_generator(test_loader, steps=None)

        best_epoch_stats['bin_test_linear_loss'] = test_loss
        best_epoch_stats['bin_test_accuracy'] = test_accuracy

        model = expt.model
        model.load_weights(ckpt_filename)

        def repeat_inference(loader, prefix='', drop_keys=[], n_times=20):
            metrics_names = [prefix + 'loss'] + [prefix + metric_name for metric_name in model.metrics_names]
            metrics_list = []

            for _ in range(n_times):
                loss, metrics = model.evaluate_generator(loader, steps=None)
                if not isinstance(metrics, np.ndarray):
                    metrics = np.array([metrics])
                metrics_list.append(np.concatenate(([loss], metrics)))
            metrics_list = [list(e) for e in zip(*metrics_list)]
            metrics_stats = pd.DataFrame({col: val for col, val in zip(metrics_names, metrics_list)})
            return metrics_stats.drop(drop_keys, axis=1, errors='ignore')

        metrics_stats = repeat_inference(train_loader, n_times=n_repetitions)

        metrics_stats = metrics_stats.join(repeat_inference(test_loader,
                                                            prefix='test_',
                                                            drop_keys=['test_bound', 'test_kl', 'test_C'],
                                                            n_times=n_repetitions))

        best_epoch_stats = best_epoch_stats.drop(metrics_stats.keys().tolist(), axis=1, errors='ignore')
        metrics_stats = metrics_stats.join(pd.concat([best_epoch_stats]*n_repetitions, ignore_index=True))

        log_filename = expt.test_log_filename
        if network in ['pbgnet_ll', 'pbcombinet_ll'] and target_metric == 'bound':
            log_filename = join(logging_path, 'bound_test_log.tsv')
        metrics_stats.to_csv(log_filename, sep='\t', index=False)

    default_irrelevant_columns = ['val_bound', 'val_kl', 'val_C']
    if network == 'pbgnet_ll':
        pbgnet_testing(target_metric='val_loss',
                       irrelevant_columns=default_irrelevant_columns,
                       n_repetitions=20)
        pbgnet_testing(target_metric='bound',
                       irrelevant_columns=default_irrelevant_columns,
                       n_repetitions=20)

    elif network == 'pbgnet':
        pbgnet_testing(target_metric='bound',
                       irrelevant_columns=['val_loss', 'val_accuracy', 'val_linear_loss'] + default_irrelevant_columns,
                       n_repetitions=20)

    elif network == 'pbcombinet_ll':
        pbgnet_testing(target_metric='val_loss',
                       irrelevant_columns=default_irrelevant_columns,
                       n_repetitions=1)
        pbgnet_testing(target_metric='bound',
                       irrelevant_columns=default_irrelevant_columns,
                       n_repetitions=1)

    elif network == 'pbcombinet':
        pbgnet_testing(target_metric='bound',
                       irrelevant_columns=['val_loss', 'val_accuracy', 'val_linear_loss'] + default_irrelevant_columns,
                       n_repetitions=1)
    if logging:
        with open(join(logging_path, 'done.txt'), 'w') as done_file:
            done_file.write("done")

    print("### DONE ###")

if __name__ == '__main__':
    launch()
