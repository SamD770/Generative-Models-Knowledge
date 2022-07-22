
import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim as optim
import numpy as np
import abc
import pytorch_lightning as pl


from modules import BernoulliFC, GaussianFC, GaussianTARNet, LikelihoodFC, PropensityScoreFC, DISTRIBUTIONS
from utils import build_fc_network


class BasicMatchVAE(pl.LightningModule):

    def __init__(self, in_dim, z_dim, encode_layer_dims_core, encode_layer_dims_branch,
                 decode_x_layer_dims, decode_t_layer_dims, likelihoods, dataset, init_lr, activation='relu',
                 dropout_prob=0., log_propensity_total_weight=2, log_propensity_treated_share=0.5):
        """
        Initialize the basic matching VAE model.

        Args:
            in_dim: Integer indicating dimension of input variable x.
            z_dim: Integer indicating dimension of latent space variable z.
            encode_layer_dims_core : List of integers indicating the number of output units of each encoding layer in the core part of the TARNet model.
            encode_layer_dims_branch : List of integers indicating the number of output units of each encoding layer in the branch parts of the TARNet model (except the last one to z).
            decode_x_layer_dims: List of integers indicating the number of output units of each decoding layer from z to x (except the last one to x).
            decode_t_layer_dims: List of integers indicating the number of output units of each decoding layer from z to t (except the last one to x).
            likelihoods: 1-d numpy array of strings denoting the likelihood model or every dimension of variable x ("cont" or "bin")
            dataset: dataset that the model is used for.
            init_lr: Initial learning rate of optimizer.
        """
        super(BasicMatchVAE, self).__init__()
        # dimensions
        self.in_dim = in_dim  # dimension of input x in forward(...)
        self.z_dim = z_dim  # dimension of latent space z
        self.encode_layer_dims_core = encode_layer_dims_core
        self.encode_layer_dims_branch = encode_layer_dims_branch
        self.decode_x_layer_dims = decode_x_layer_dims
        self.decode_t_layer_dims = decode_t_layer_dims
        if isinstance(likelihoods, str):
            likelihoods = np.array([likelihoods] * in_dim)
        self.likelihoods = likelihoods
        self.dataset = dataset
        self.init_lr = init_lr

        self.masks = {}
        for lkl in DISTRIBUTIONS.keys():
            self.masks[lkl] = (likelihoods == lkl).ravel()

        self.log_propensity_total_weight = log_propensity_total_weight
        self.log_propensity_treated_share = log_propensity_treated_share

        # prior params
        self.mu_p_z = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        # TODO extend to full-covariance case
        self.log_sigma_square_p_z = nn.Parameter(torch.zeros(z_dim), requires_grad=False)  # logits

        # recognition model
        self.encoder = GaussianTARNet(in_dim=in_dim, out_dim=z_dim, dropout_prob=dropout_prob, activation=activation,
                                      layer_dims_core=encode_layer_dims_core,
                                      layer_dims_branch=encode_layer_dims_branch).to(self.device)

        # generative model
        self.decoder_x = LikelihoodFC(in_dim=z_dim, out_dim=in_dim, layer_dims=decode_x_layer_dims,
                                      likelihoods=likelihoods, dropout_prob=dropout_prob,
                                      activation=activation).to(self.device)
        self.decoder_t = PropensityScoreFC(in_dim=z_dim, out_dim=1, layer_dims=decode_t_layer_dims,
                                           dropout_prob=dropout_prob, activation=activation).to(self.device)

    @property
    def sigma_square_p_z(self):
        return torch.exp(self.log_sigma_square_p_z)

    def forward(self, x, t):
        """
        Pass x, t through the basic matching VAE network (x -> x_hat).
        Passes x, t through the encoder, samples from all q(z | x, t) and passes z sample through the decoder to obtain parameters of p(x,t|z).

        Args:
            x: A tensor of dimension (self.in_dim).
            t: A tensor of dimension (1).
        Returns:
            params_p_x_z: The parameters of p(x | z), a tensor of dimension (self.in_dim).
            params_p_t_z: The parameters of p(t | z), a tensor of dimension (1).
            q_z_xt: List of J Gaussian distribution objects.
            z_sample_q_z_xt: List of z samples.
        """
        mu_q_z_j_xt, log_sigma_square_q_z_j_xt = self.encode(x, t)
        q_z_xt = D.Independent(D.Normal(loc=mu_q_z_j_xt, scale=torch.sqrt(torch.exp(log_sigma_square_q_z_j_xt))), 1)
        z_sample_q_z_xt = q_z_xt.rsample()
        params_p_x_z, params_p_t_z = self.decode(z_sample_q_z_xt)

        return params_p_x_z, params_p_t_z, q_z_xt, z_sample_q_z_xt

    def encode(self, x, t):
        """
        Estimate parameters of q(z | x, t), and sample from these distribution.

        Args:
            x: A tensor of dimension (self.in_dim).
            t: A tensor of dimension (1).
        Returns:
            mu and log(variance) of q(z | x, t), each tensor of dimension (self.z_dim).
        """
        mu_q_z_xt, log_sigma_square_q_z_xt = self.encoder(x, t)

        return mu_q_z_xt, log_sigma_square_q_z_xt

    def decode(self, z_sample_q_z_xt):
        """
        Estimate the parameters of p(x | z).

        Args:
            z_sample_q_z_xt: z sample, tensor of dimension (self.z_dim).
        Returns:
            p_x_z_params : Parameters of p(x | z), a tensor of dimension (self.in_dim).
            p_t_z_params : Parameters of p(t | z), a tensor of dimension (1).
        """
        # TODO implement custom likelihood models with masking (see HI-VAE)
        p_x_z_params = self.decoder_x(z_sample_q_z_xt)
        p_t_z_params = self.decoder_t(z_sample_q_z_xt)

        return p_x_z_params, p_t_z_params

    def compute_loss(self, x, t, params_p_x_z, params_p_t_z, q_z_xt, z_sample_q_z_xt):
        """
        Computes the ELBO of the log likelihood, with a negative sign (-> loss).

        Assumes L=1 (1 MC sample drawn).

        For L>1, all arguments of compute_loss_new would have to have one more dimension which is the l dimension
        (-> we have to sample z multiple times, need multiple x batches etc.).

        Args:
            x: A tensor of dimension (self.in_dim).
            t: A tensor of dimension (1).
            params_p_x_z: The parameters of p(x | z), a tensor of dimension (self.in_dim).
            params_p_t_z: The parameters of p(t | z), a tensor of dimension (1).
            q_z_xt: Normal distribution object of q(z | x, t).
            z_sample_q_z_xt: Samples drawn from q(z | x, t).
        Returns:
            The average loss for this batch.
        """

        p_x_z = {lkl: D.Independent(DISTRIBUTIONS[lkl](**params_p_x_z[lkl]), 1) for lkl in DISTRIBUTIONS.keys() if
                 lkl in params_p_x_z}

        x_sep_by_lkl = {lkl: x[..., self.masks[lkl]] for lkl in p_x_z.keys()}
        log_prob_p_x_z = sum([p_x_z[lkl].log_prob(x_sep_by_lkl[lkl]) for lkl in p_x_z.keys()]).ravel()


        p_t_z = D.Independent(D.Bernoulli(probs=torch.clamp(params_p_t_z, min=1e-10, max=1 - (1e-10))), 1)
        log_prob_p_t_z = p_t_z.log_prob(t).ravel()
        propensity_weights = self.log_propensity_total_weight * (
                    self.log_propensity_treated_share * t + (1 - self.log_propensity_treated_share) * (1 - t)).ravel()
        p_z = D.Independent(D.Normal(loc=self.mu_p_z, scale=torch.sqrt(self.sigma_square_p_z)), 1)
        kl_z = D.kl.kl_divergence(q_z_xt, p_z).ravel()

        ELBO = torch.mean(log_prob_p_x_z + propensity_weights * log_prob_p_t_z - kl_z)  # average over mini-batch
        loss = -ELBO

        return loss, log_prob_p_x_z, log_prob_p_t_z, kl_z

    def training_step(self, batch, batch_idx):
        if self.dataset in ["news", "synthetic1"]:
            x, y, t, mu0, mu1, cate_true = batch

        params_p_x_z, params_p_t_z, q_z_xt, z_sample_q_z_xt = self.forward(x, t)
        loss, _, _, kl_z = self.compute_loss(x, t, params_p_x_z, params_p_t_z, q_z_xt, z_sample_q_z_xt)

        self.log('train_metric/loss', loss, on_step=False, on_epoch=True)
        self.log('train_metric/kl_z', kl_z, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.dataset in ["news", "synthetic1"]:
            x, y, t, mu0, mu1, cate_true = batch

        params_p_x_z, params_p_t_z, q_z_xt, z_sample_q_z_xt = self.forward(x, t)
        loss, _, _, kl_z = self.compute_loss(x, t, params_p_x_z, params_p_t_z, q_z_xt, z_sample_q_z_xt)

        self.log('val_metric/loss', loss, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr)

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer





class PlainVAE(pl.LightningModule):

    def __init__(self, in_dim, z_dim, encode_layer_dims, decode_layer_dims, likelihoods, dataset, init_lr, activation='relu',dropout_prob=0.):
        """
        Initialize the plain VAE model.

        Args:
            in_dim: Integer indicating dimension of input variable x.
            z_dim: Integer indicating dimension of latent space variable z.
            encode_layer_dims: List of integers indicating the number of output units of each encoding layer (except the last one to z).
            decode_layer_dims: List of integers indicating the number of output units of each decoding layer (except the last one to x).
            likelihoods: 1-d numpy array of strings denoting the likelihood model or every dimension of variable x ("cont" or "bin")
            dataset: dataset that the model is used for.
            init_lr: Initial learning rate of optimizer.
        """
        super(PlainVAE, self).__init__()
        # dimensions
        self.in_dim = in_dim  # dimension of input x in forward(...)
        self.z_dim = z_dim  # dimension of latent space z
        self.encode_layer_dims = encode_layer_dims
        self.decode_layer_dims = decode_layer_dims
        if isinstance(likelihoods, str):
            likelihoods = np.array([likelihoods] * in_dim)
        self.likelihoods = likelihoods
        self.dataset = dataset
        self.init_lr = init_lr

        self.masks = {}
        for lkl in DISTRIBUTIONS.keys():
            self.masks[lkl] = (likelihoods == lkl).ravel()


        # prior params
        self.mu_p_z = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        # TODO extend to full-covariance case
        self.log_sigma_square_p_z = nn.Parameter(torch.zeros(z_dim), requires_grad=False)  # logits

        # recognition model
        self.encoder = GaussianFC(in_dim=in_dim, out_dim=z_dim, layer_dims=encode_layer_dims, dropout_prob=dropout_prob, activation=activation).to(self.device)

        # generative model
        self.decoder = LikelihoodFC(in_dim=z_dim, out_dim=in_dim, layer_dims=decode_layer_dims, likelihoods=likelihoods, dropout_prob=dropout_prob, activation=activation).to(self.device)

    @property
    def sigma_square_p_z(self):
        return torch.exp(self.log_sigma_square_p_z)

    def forward(self, x):
        """
        Pass x through the VaDE network (x -> x_hat).
        Passes x through the encoder, samples from all q(z | x) and passes z sample through the decoder to obtain parameters of p(x|z).

        Args:
            x: The input of the network, a tensor of dimension (self.in_dim).
        Returns:
            p_x_z_params: The parameters of p(x | z), a tensor of dimension (self.in_dim).
            q_z_x: List of J Gaussian distribution objects.
            z_sample_q_z_x: List of z samples.
        """
        mu_q_z_j_x, log_sigma_square_q_z_j_x = self.encode(x)
        q_z_x = D.Independent(D.Normal(loc=mu_q_z_j_x, scale=torch.sqrt(torch.exp(log_sigma_square_q_z_j_x))), 1)
        z_sample_q_z_x = q_z_x.rsample()
        params_p_x_z = self.decode(z_sample_q_z_x)

        return params_p_x_z, q_z_x, z_sample_q_z_x

    def encode(self, x):
        """
        Estimate parameters of q(z | x), and sample from these distribution.

        Args:
            x: Input tensor of dimension (self.in_dim).

        Returns:
            mu and log(variance) of q(z | x), each tensor of dimension (self.z_dim).
        """
        mu_q_z_x, log_sigma_square_q_z_x = self.encoder(x)

        return mu_q_z_x, log_sigma_square_q_z_x

    def decode(self, z_sample_q_z_x):
        """
        Estimate the parameters of p(x | z).

        Args:
            z_sample_q_z_x: z sample, tensor of dimension (self.z_dim).
        Returns:
            x: Parameters of p(x | z), a tensor of dimension (self.in_dim).
        """
        # TODO implement custom likelihood models with masking (see HI-VAE)
        p_x_z_params = self.decoder(z_sample_q_z_x)

        return p_x_z_params

    def compute_loss(self, x, params_p_x_z, q_z_x, z_sample_q_z_x):
        """
        Computes the ELBO of the log likelihood, with a negative sign (-> loss).

        Assumes L=1 (1 MC sample drawn).

        For L>1, all arguments of compute_loss_new would have to have one more dimension which is the l dimension
        (-> we have to sample z multiple times, need multiple x batches etc.).

        Args:
            x: The input of the network, a tensor of dimension (self.in_dim).
            params_p_x_z: The parameters of p(x | z), a tensor of dimension (self.in_dim).
            q_z_x: Normal distribution object of q(z | x).
            z_sample_q_z_x: Samples drawn from q(z | x).
        Returns:
            The average loss for this batch.
        """

        p_x_z = {lkl: D.Independent(DISTRIBUTIONS[lkl](**params_p_x_z[lkl]), 1) for lkl in DISTRIBUTIONS.keys() if
                 lkl in params_p_x_z}

        x_sep_by_lkl = {lkl: x[..., self.masks[lkl]] for lkl in p_x_z.keys()}
        log_prob_p_x_z = sum([p_x_z[lkl].log_prob(x_sep_by_lkl[lkl]) for lkl in p_x_z.keys()]).ravel()

        p_z = D.Independent(D.Normal(loc=self.mu_p_z, scale=torch.sqrt(self.sigma_square_p_z)), 1)
        kl_z = D.kl.kl_divergence(q_z_x, p_z)

        ELBO = torch.mean(log_prob_p_x_z - kl_z)  # average over mini-batch
        loss = -ELBO

        return loss, log_prob_p_x_z, kl_z


    def training_step(self, batch, batch_idx):
        if self.dataset in ["news", "synthetic1"]:
            x, y, t, mu0, mu1, cate_true = batch

        params_p_x_z, q_z_x, z_sample_q_z_x = self.forward(x)
        loss, _, kl_z = self.compute_loss(x, params_p_x_z, q_z_x, z_sample_q_z_x)

        self.log('train_metric/loss', loss, on_step=False, on_epoch=True)
        self.log('train_metric/kl_z', kl_z, on_step=False, on_epoch=True)

        return loss



    def validation_step(self, batch, batch_idx):
        if self.dataset in ["news", "synthetic1"]:
            x, y, t, mu0, mu1, cate_true = batch

        params_p_x_z, q_z_x, z_sample_q_z_x = self.forward(x)
        loss, _, kl_z = self.compute_loss(x, params_p_x_z, q_z_x, z_sample_q_z_x)

        self.log('val_metric/loss', loss, on_step=False, on_epoch=True)

        return loss


    def configure_optimizers(self):
        # define optimizer and scheduler
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.init_lr)

        # training and evaluation loop
        epoch_lr = optimizer.param_groups[0]['lr']

        # TODO add support for learning rate scheduler
        # adjust learning rate
        # if epoch % args.update_lr_every_epoch == 0 and not epoch == 0:
        #     adjust_learning_rate(optimizer, epoch_lr, args.lr_decay, args.min_lr)

        return optimizer


# TODO initialization of params with glorot
