import torch
import torch.distributions as D
import torch.nn as nn
import numpy as np

DISTRIBUTIONS = {
    "cont": D.Normal,
    "bin": D.Bernoulli,
    "poisson": D.Poisson
}


def build_fc_network(layer_dims, activation="relu", dropout_prob=0.):
    """
    Stacks multiple fully-connected layers with an activation function and a dropout layer in between.

    - Source used as orientation: https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py

    Args:
        layer_dims: A list of integers, where (starting from 1) the (i-1)th and ith entry indicates the input
                    and output dimension of the ith layer, respectively.
        activation: Activation function to choose. "relu" or "sigmoid".
        dropout_prob: Dropout probability between every fully connected layer with activation.

    Returns:
        An nn.Sequential object of the layers.
    """
    # Note: possible alternative: OrderedDictionary
    net = []
    for i in range(1, len(layer_dims)):
        net.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "elu":
            net.append(nn.ELU())
        net.append(nn.Dropout(dropout_prob))
    net = nn.Sequential(*net)  # unpacks list as separate arguments to be passed to function

    return net


class LikelihoodFC(nn.Module):
    def __init__(self, in_dim, layer_dims, out_dim, likelihoods=[], activation="relu", dropout_prob=0.):

        assert (len(likelihoods) == out_dim)
        self.masks = {}
        self.lengths = {}
        for lkl in DISTRIBUTIONS.keys():
            self.masks[lkl] = (likelihoods == lkl).ravel()
            self.lengths[lkl] = int((likelihoods == lkl).sum())

        super(LikelihoodFC, self).__init__()

        self.core = build_fc_network([in_dim] + layer_dims, activation=activation,
                                     dropout_prob=dropout_prob)  # most layers of the model
        self.lkl_head_activations = {
            "cont": {"loc": None, "scale": (lambda s: torch.sqrt(torch.exp(s)))},
            "bin": {"logits": None},
            "poisson": {"rate": (lambda s: torch.clamp(torch.exp(s), min=1e-10))},
        }

        self.lkl_head_layers = nn.ModuleDict({
            lkl: nn.ModuleDict({param: nn.Linear(layer_dims[-1], self.lengths[lkl]) for param in
                                self.lkl_head_activations[lkl].keys()}) for lkl in self.lkl_head_activations.keys()
        })
        assert len([lkl for lkl in self.lkl_head_activations.keys() if lkl not in DISTRIBUTIONS]) == 0

    def forward(self, x):
        h = self.core(x)
        params = {}
        for lkl in self.lkl_head_layers.keys():
            params[lkl] = {param: self.lkl_head_layers[lkl][param](h) for param in self.lkl_head_layers[lkl].keys()}
            for param in params[lkl].keys():
                if self.lkl_head_activations[lkl][param] is not None:
                    params[lkl][param] = self.lkl_head_activations[lkl][param](params[lkl][param])

        return params


class GaussianFC(nn.Module):

    def __init__(self, in_dim, layer_dims, out_dim, activation = "relu", dropout_prob = 0.):

        super(GaussianFC, self).__init__()

        self.core = build_fc_network([in_dim] + layer_dims, activation=activation, dropout_prob=dropout_prob)  # most layers of the model
        self.fc_mu = nn.Linear(layer_dims[-1],
                               out_dim)  # layer that outputs the mean parameter of the Gaussian distribution ; no activation function, since on continuous scale
        self.fc_log_sigma = nn.Linear(layer_dims[-1],
                                      out_dim)  # layer that outputs the logarithm of the Gaussian distribution (diagonal covariance structure); no activation function, since on continuous scale


    def forward(self, x):
        h = self.core(x)
        mu = self.fc_mu(h)
        log_sigma_square = self.fc_log_sigma(h)

        return mu, log_sigma_square
