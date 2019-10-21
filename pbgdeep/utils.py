import math
import torch

from poutyne.framework.metrics.epoch_metrics import EpochMetric

def linear_loss(pred_y, y):
    """Linear loss function."""
    return torch.mean((1 -(y * pred_y)) / 2)

def accuracy(pred_y, y):
    """Binary classification accuracy function."""
    pred_y_copy = pred_y.clone()
    pred_y_copy[pred_y_copy < 0] = -1
    pred_y_copy[pred_y_copy >= 0] = 1
    acc_pred = (pred_y_copy == y).float().sum() / y.shape[0]
    return acc_pred * 100


def bound(linear_loss, kl, delta, n_examples, C):
    """Bound computation as presented in Theorem 3. for given C values."""
    bound_value = (1 / (1 - torch.exp(-C))) * \
                  (1 - torch.exp(-C * linear_loss - (1 / n_examples) * \
                  (kl + math.log(2 * math.sqrt(n_examples) / delta))))

    return bound_value

def get_logging_dir_name(experiment_setting):
    """Map experiment config dictionnary to a unique directory name."""
    return f"{experiment_setting['network']}_H{experiment_setting['hidden_layers']}-{experiment_setting['hidden_size']}"\
            + f"_B{experiment_setting['batch_size']}_{experiment_setting['optim_algo']}_WD{experiment_setting['weight_decay']}"\
            + f"_LR{experiment_setting['learning_rate']}_P{experiment_setting['lr_patience']}"\
            + f"_V{experiment_setting['valid_size']}_R{experiment_setting['random_seed']}"\
            + (f"_S{experiment_setting['sample_size']}_{experiment_setting['prior']}" if experiment_setting['network'].startswith('pbg') else "")\
            + (f"_PE{experiment_setting['pre_epochs']}" if experiment_setting['prior'] == 'pretrain' else "")

class MasterMetricLogger(EpochMetric):
    """Computes all bound related epoch metrics (linear loss, kl, C, bound)."""

    def __init__(self, network, loss_function, delta, n_examples, C_range=None):
        super().__init__()
        self.network = network
        self.loss_function = loss_function
        self.delta = delta
        self.n_examples = n_examples
        self.C_range = C_range
        self.__name__ = self.loss_function.__name__
        self._reset()

    def forward(self, y_prediction, y_true):
        self.loss_sum += self.loss_function(y_prediction, y_true) * y_true.shape[0]
        self.example_sum += y_true.shape[0]

    def _reset(self):
        self.loss_sum = 0
        self.example_sum = 0

    def get_metric(self):
        loss = self.loss_sum / self.example_sum
        with torch.no_grad():
            C = torch.exp(self.network.t) if self.C_range is None else self.C_range
            kl = self.network.compute_kl()

            bound_value = bound(loss, kl, self.delta, self.n_examples, C)
        metrics = {'linear_loss': loss.item(), 'kl': kl.item()}
        if self.C_range is None:
            metrics.update({'C': C.item(), 'bound': bound_value.item()})
        else:
            min_idx = bound_value.argmin()
            metrics.update({'C': C[min_idx].item(), 'bound': bound_value[min_idx].item()})
        self.network.metrics.update(metrics)
        self._reset()
        return loss

class MetricLogger(EpochMetric):
    """Logs a single epoch metric (key) computed by the MasterMetricLogger."""

    def __init__(self, network, key):
        super().__init__()
        self.network = network
        self.key = key
        self.__name__ = key

    def forward(self, y_prediction, y_true):
        pass

    def get_metric(self):
        return self.network.metrics[self.key]
