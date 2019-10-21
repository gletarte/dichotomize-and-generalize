import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter

from pbgdeep.utils import linear_loss, bound

class PBGInputFunction(Function):
    """PBGNet input layer function."""
    @staticmethod
    def forward(ctx, input, weight, bias, sample_size, training):
        linear = F.linear(input, weight, bias)
        m = linear / (math.sqrt(2) * torch.norm(input, dim=1, keepdim=True))

        if training:
            # Computing and saving gradient
            exp_value = torch.exp(-(linear / (math.sqrt(2) * torch.norm(input, dim=1, keepdim=True))) ** 2)
            grad_b = exp_value * (math.sqrt(2) / (math.sqrt(math.pi) * torch.norm(input, dim=1, keepdim=True)))
            grad_w = exp_value.unsqueeze(1) * ((input.unsqueeze(-1) * math.sqrt(2)) / (math.sqrt(math.pi) * torch.norm(input, dim=1, keepdim=True)).unsqueeze(-1))
            ctx.save_for_backward(grad_w, grad_b)

        return torch.erf(m)

    @staticmethod
    def backward(ctx, grad_output):
        grad_w, grad_b = ctx.saved_tensors

        # Backpropagation
        grad_w = ((grad_output.unsqueeze(1) * grad_w).sum(0)).t()
        grad_b = (grad_output * grad_b).sum(0)

        return None, grad_w, grad_b, None, None

class PBGHiddenFunction(Function):
    """PBGNet hidden layer function."""
    @staticmethod
    def forward(ctx, input, weight, bias, sample_size, training):
        proba = 0.5 + 0.5 * input
        s = proba.unsqueeze(-1).expand(*proba.shape, sample_size)
        s = 2 * torch.bernoulli(s) - 1

        linear = F.linear(s.transpose(1, 2), weight, bias)
        erf_value = torch.erf(linear / (math.sqrt(2 * input.shape[1])))
        norm_term = 1 / sample_size

        if training:
            # Computing and saving gradient
            exp_value = torch.exp(-(linear / (math.sqrt(2 * input.shape[1]))) ** 2)
            grad_b = exp_value * (math.sqrt(2) / (math.sqrt(math.pi * input.shape[1])))
            grad_w = ((s * math.sqrt(2)) / (math.sqrt(math.pi * input.shape[1]))).matmul(exp_value)

            grad_w = norm_term * grad_w
            grad_b = norm_term * grad_b.sum(1)

            p_s = 0.5 + 0.5 * input.unsqueeze(-1) * s
            grad_input = (s / (2 * p_s)).matmul(erf_value)
            grad_input = norm_term * grad_input
            ctx.save_for_backward(grad_input, grad_w, grad_b)

        return norm_term * erf_value.sum(1)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_w, grad_b = ctx.saved_tensors

        # Backpropagation
        grad_w = ((grad_output.unsqueeze(1) * grad_w).sum(0)).t()
        grad_b = (grad_output * grad_b).sum(0)
        grad_input = (grad_output.unsqueeze(1) * grad_input).sum(-1)

        return grad_input, grad_w, grad_b, None, None

class PBGOutputFunction(Function):
    """PBGNet output layer function."""

    @staticmethod
    def forward(ctx, input, weight, bias, sample_size, training):
        proba = 0.5 + 0.5 * input
        s = proba.unsqueeze(-1).expand(*proba.shape, sample_size)
        s = 2 * torch.bernoulli(s) - 1

        linear = F.linear(s.transpose(1, 2), weight, bias)
        erf_value = torch.erf(linear / (math.sqrt(2 * input.shape[1]))).squeeze(-1)
        norm_term = 1 / sample_size

        if training:
            # Computing and saving gradient
            exp_value = torch.exp(-(linear / (math.sqrt(2 * input.shape[1]))) ** 2).squeeze()
            grad_b = exp_value.unsqueeze(1) * (math.sqrt(2) / (math.sqrt(math.pi * input.shape[1])))
            grad_w = ((s * math.sqrt(2)) / (math.sqrt(math.pi * input.shape[1]))).matmul(exp_value.unsqueeze(-1)).squeeze()

            grad_w = norm_term * grad_w
            grad_b = norm_term * grad_b.sum(-1)

            p_s = 0.5 + 0.5 * input.unsqueeze(-1) * s
            grad_input = (s / (2 * p_s)).matmul(erf_value.unsqueeze(-1)).squeeze()
            grad_input = norm_term * grad_input
            ctx.save_for_backward(grad_input, grad_w, grad_b)

        return norm_term * erf_value.sum(-1).unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_w, grad_b = ctx.saved_tensors

        # Backpropagation
        grad_w = (grad_output * grad_w).sum(0).unsqueeze(0)
        grad_b = (grad_output * grad_b).sum(0)
        grad_input = grad_output * grad_input

        return grad_input, grad_w, grad_b, None, None

class PBGLayer(torch.nn.Module):
    """PAC-Bayesian Binary Gradient Network architecture layer module.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        position (str): Position of layer in PBGNet architecture (either 'input', 'hidden' or 'output')
        sample_size (int): Sample size T for Monte Carlo approximation (Default value = 100).
    """

    def __init__(self, in_features, out_features, position, sample_size=100):
        super(PBGLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.position = position
        self.sample_size = sample_size

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.priors = torch.nn.ParameterDict({'weight': Parameter(torch.Tensor([0.0]), requires_grad=False),
                                              'bias': Parameter(torch.Tensor([0.0]), requires_grad=False)})

        if self.position == 'input':
            self.forward_fct = PBGInputFunction
        elif self.position == 'hidden':
            self.forward_fct = PBGHiddenFunction
        else:
            self.forward_fct = PBGOutputFunction


    def forward(self, input):
        return self.forward_fct.apply(input, self.weight, self.bias, self.sample_size, self.training)

    def __repr__(self):
        return self.__class__.__name__ + '_' + self.position + ' (' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

class PBGNet(torch.nn.Module):
    """PAC-Bayesian Binary Gradient Network architecture (stochastic approximation) PyTorch module.

    Args:
        input_size (int): Input data point dimension d_0.
        hidden_layers (list): List of number of neurons per layers from first layer d_1 to before last layer d_{L-1}.
        n_examples (int): Number of examples in the training set, used for bound computation.
        sample_size (int): Sample size T for Monte Carlo approximation (Default value = 100).
        delta (float): Delta parameter of PAC-Bayesian bounds, see Theorem 1 (Default value = 0.05).
    """

    def __init__(self, input_size, hidden_layers, n_examples, sample_size=64, delta=0.05):
        super(PBGNet, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.n_examples = n_examples
        self.sample_size = sample_size
        self.delta = delta
        self.t = Parameter(torch.Tensor(1))
        self.metrics = {}

        self.layers = torch.nn.ModuleList()
        self.layers.append(PBGLayer(self.input_size, self.hidden_layers[0], 'input', sample_size))
        for in_dim, out_dim in zip(self.hidden_layers[:-1], self.hidden_layers[1:]):
            self.layers.append(PBGLayer(in_dim, out_dim, 'hidden', sample_size))
        self.layers.append(PBGLayer(self.hidden_layers[-1], 1, 'output', sample_size))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

    def bound(self, pred_y, y):
        """Bound computation as presented in Theorem 3. with the learned C value."""
        loss = linear_loss(pred_y, y)

        C = torch.exp(self.t)
        kl = self.compute_kl()
        bound_value = bound(loss, kl, self.delta, self.n_examples, C)
        return bound_value

    def compute_kl(self):
        """Kullback-Leibler divergence computation as presented in equation 17."""
        kl = 0
        coefficient = 1
        for i, layer in zip(itertools.count(len(self.layers) -1, -1), reversed(self.layers)):
            norm = torch.norm(layer.weight - layer.priors['weight']) ** 2 + \
                                 torch.norm(layer.bias - layer.priors['bias']) ** 2
            kl += coefficient * norm
            coefficient *= layer.out_features
        return 0.5 * kl

    def set_priors(self, state_dict):
        """Sets network parameters priors."""
        for i, layer in enumerate(self.layers):
            layer.priors['weight'] = Parameter(state_dict[f"layers.{i}.weight"].data.clone(), requires_grad=False)
            layer.priors['bias'] = Parameter(state_dict[f"layers.{i}.bias"].data.clone(), requires_grad=False)

    def init_weights(self):
        """Network parameters random initialization."""
        self.t.data.zero_()
        for layer in self.layers:
            if isinstance(layer, PBGLayer):
                torch.nn.init.xavier_normal_(layer.weight)
                layer.bias.data.zero_()

    def set_sample_size(self, sample_size):
        """Allows to modify the sample size at any given time."""
        self.sample_size = sample_size
        for layer in self.layers:
            layer.sample_size = sample_size

class BaselineNet(torch.nn.Module):
    """Standard neural network architecture used as a baseline.

    Args:
        input_size (int): Input data point dimension d_0.
        hidden_layers (list): List of number of neurons per layers from first layer d_1 to before last layer d_{L-1}.
        activation_fct (function): Activation function.
    """

    def __init__(self, input_size, hidden_layers, activation_fct):
        super(BaselineNet, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation_fct = activation_fct

        self.layers = []
        for in_dim, out_dim in zip([self.input_size, *self.hidden_layers], [*self.hidden_layers, 1]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(self.activation_fct())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.layers(input)

    def init_weights(self):
        """Network parameters random initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                layer.bias.data.zero_()

class PBCombiBaseLayer(torch.nn.Module):
    """PAC-Bayesian Binary Gradient Deterministic Network architecture base layer module.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
    """
    def __init__(self, in_features, out_features):
        super(PBCombiBaseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.priors = torch.nn.ParameterDict({'weight': Parameter(torch.Tensor([0.0]), requires_grad=False),
                                              'bias': Parameter(torch.Tensor([0.0]), requires_grad=False)})

    def forward(self, input):
        raise NotImplementedError()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

class PBCombiInputLayer(PBCombiBaseLayer):
    """PBCombiNet input layer."""
    def __init__(self, in_features, out_features):
        super(PBCombiInputLayer, self).__init__(in_features, out_features)

    def forward(self, input):
        m = (F.linear(input, self.weight, self.bias)) / (math.sqrt(2)*torch.norm(input, dim=1, keepdim=True))
        return torch.erf(m)

class PBCombiHiddenLayer(PBCombiBaseLayer):
    """PBCombiNet hidden layer."""
    def __init__(self, in_features, out_features):
        super(PBCombiHiddenLayer, self).__init__(in_features, out_features)
        self.register_buffer('s', torch.FloatTensor(list(itertools.product([-1, 1], repeat=self.in_features))))

    def forward(self, input):
        p_s = 0.5 + 0.5 * input.unsqueeze(-1) * self.s.t().unsqueeze(0)
        p_s_m = p_s.prod(dim=1)
        erf_value = torch.erf(F.linear(self.s, self.weight, self.bias)/(math.sqrt(2*self.in_features))).squeeze()

        return p_s_m.matmul(erf_value)

class PBCombiOutputLayer(PBCombiBaseLayer):
    """PBCombiNet output layer."""
    def __init__(self, in_features, out_features):
        super(PBCombiOutputLayer, self).__init__(in_features, out_features)
        self.register_buffer('s', torch.FloatTensor(list(itertools.product([-1, 1], repeat=self.in_features))))

    def forward(self, input):
        p_s = 0.5 + 0.5 * input.unsqueeze(-1) * self.s.t().unsqueeze(0)
        p_s_m = p_s.prod(dim=1)
        erf_value = torch.erf(F.linear(self.s, self.weight, self.bias)/(math.sqrt(2*self.in_features))).squeeze()

        out = p_s_m.matmul(erf_value)
        return out.unsqueeze(-1) if len(out.shape) == 1 else out

class PBCombiNet(torch.nn.Module):
    """PAC-Bayesian Binary Gradient Deterministic Network architecture (combinatorial sum) PyTorch module.

    Args:
        input_size (int): Input data point dimension d_0.
        hidden_layers (list): List of number of neurons per layers from first layer d_1 to before last layer d_{L-1}.
        n_examples (int): Number of examples in the training set, used for bound computation.
        delta (float): Delta parameter of PAC-Bayesian bounds, see Theorem 1 (Default value = 0.05).
    """

    def __init__(self, input_size, hidden_layers, n_examples, delta=0.05):
        super(PBCombiNet, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.n_examples = n_examples
        self.delta = delta
        self.t = Parameter(torch.Tensor(1))
        self.metrics = {}

        self.layers = torch.nn.ModuleList()
        self.layers.append(PBCombiInputLayer(self.input_size, self.hidden_layers[0]))
        for in_dim, out_dim in zip(self.hidden_layers[:-1], self.hidden_layers[1:]):
            self.layers.append(PBCombiHiddenLayer(in_dim, out_dim))
        self.layers.append(PBCombiOutputLayer(self.hidden_layers[-1], 1))

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

    def bound(self, pred_y, y):
        """Bound computation as presented in Theorem 3. with the learned C value."""
        loss = linear_loss(pred_y, y)
        C = torch.exp(self.t)
        kl = self.compute_kl()
        bound_value = bound(loss, kl, self.delta, self.n_examples, C)
        return bound_value

    def compute_kl(self):
        """Kullback-Leibler divergence computation as presented in equation 17."""
        kl = 0
        coefficient = 1
        for i, layer in zip(itertools.count(len(self.layers) -1, -1), reversed(self.layers)):
            norm = torch.norm(layer.weight - layer.priors['weight']) ** 2 + \
                                 torch.norm(layer.bias - layer.priors['bias']) ** 2
            kl += coefficient * norm
            coefficient *= layer.out_features
        return 0.5 * kl

    def set_priors(self, state_dict):
        """Sets network parameters priors."""
        for i, layer in enumerate(self.layers):
            layer.priors['weight'] = Parameter(state_dict[f"layers.{i}.weight"].data.clone(), requires_grad=False)
            layer.priors['bias'] = Parameter(state_dict[f"layers.{i}.bias"].data.clone(), requires_grad=False)

    def init_weights(self):
        """Network parameters random initialization."""
        self.t.data.zero_()
        for layer in self.layers:
            if isinstance(layer, PBCombiBaseLayer):
                torch.nn.init.xavier_normal_(layer.weight)
                layer.bias.data.zero_()
