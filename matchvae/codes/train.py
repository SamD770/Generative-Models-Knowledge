
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.distributions as D


from torch.autograd import Variable

import numpy as np
import os
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer, seed_everything
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import pickle

torch.backends.cudnn.benchmark = True  # for potential speedup, see https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/ (6.)

from utils import str2bool, adjust_learning_rate
from datasets import News, Synthetic1
from models import PlainVAE
from models import BasicMatchVAE
from matching import matching_metrics
from datetime import datetime
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import json


# avoid weird error
torch.set_num_threads(1)

def train_args_parser():

    parser = argparse.ArgumentParser(description='MatchVAE training')

    # general
    parser.add_argument('--device', type=str, default="cuda", metavar='N',  # e.g. "cuda", "cpu", ...
                        help='device to use for all heavy tensor operations')
    parser.add_argument('--wandb_mode', type=str, default="online", metavar='N',
                        help="mode of wandb run tracking, either no tracking ('disabled') or with tracking ('online')")
    parser.add_argument('--wandb_dir', type=str, default="../outputs", metavar='N',
                        help="directory where the wandb folder is created, default same as code.")

    parser.add_argument('--wandb_config', type=str, default='wandb_config.json', metavar='N',
                        help="wandb config file")
    parser.add_argument('--model_type', type=str, default='basic_match_vae', metavar='N',
                        help="model type to use, EITHER 'plain_vae' (plain VAE as in Kingma, 2016) OR 'match_vae' (our idea)")
    parser.add_argument('--dataset', type=str, default='synthetic1', metavar='N',
                        help="dataset used during training, one in ['news', 'synthetic1']; \
                                  note that the fast_... variants require dataset_on_gpu=True")
    parser.add_argument('--data_folder', type=str, default='../data', metavar='N',
                        help="data folder, default is assumed to be ./data/")

    # TODO can this be done in a nicer way?
    parser.add_argument('--dataset_on_gpu', type=str2bool, nargs='?', dest='dataset_on_gpu', const=True, default=True,
                        # special parsing of boolean argument
                        help='whether the dataset is on GPU or not (batches are loaded separately to GPU)')
    parser.add_argument('--save_model', type=str2bool, nargs='?', dest='save_model', const=True, default=True,
                        help='whether to save the model or not')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--config_args_path', type=str, default="", metavar='N',
                        # e.g. "/home/cdt/vadeplusplus/wandb/run-20210216_164038-qy5spi0g/files/args_dict.pickle", "config/args_dict.pickle"
                        help="the path to the args config dictionary to be loaded. If a path is provided, all specifications of hyperparameters above are ignored. \
                             If the argument is an empty string, the hyperparameter specifications above are used as usual.")
    parser.add_argument('--do_val_during_training', type=str2bool, nargs='?', dest='do_val_during_training', const=True,
                        default=True,  # special parsing of boolean argument
                        help='whether to perform evaluation on the test set throughout training, if True, do it for every epoch, otherwise only do it at the end of the training')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, metavar='N',
                        help='every how many epochs to perform a validation')

    parser.add_argument('--z_dim', type=int, default=20, metavar='N',
                        help='dimension of z latent variable')
    parser.add_argument('--dropout_prob', type=float, default=0., metavar='N')
    parser.add_argument('--activation', type=str, default="elu", metavar='N')
    # TODO : refactor the model chocie
    parser.add_argument('--n_layers_core', type=int, default=3,
                        help="number of layers in the core part.")
    parser.add_argument('--n_layers_branch', type=int, default=3,
                        help="number of layers in the branch part.")
    parser.add_argument('--n_hidden', type=int, default=200,
                        help="number of hidden units in each layer.")

    parser.add_argument('--n_layers_t', type=int, default=4,
                        help="number of layers in the core part.")
    parser.add_argument('--n_hidden_t', type=int, default=2,
                        help="number of hidden units.")
    parser.add_argument('--n_hidden_t_increment', type=int, default=1,
                        help="increment in the number of hidden units.")


    # PlainVAE
    parser.add_argument('--encode_layer_dims', type=int, nargs='+', default=None,
                        metavar='N',
                        help="list of integers, each indicating the output dimensions of each hidden layer.")
    parser.add_argument('--decode_layer_dims', type=int, nargs='+', default=None, metavar='N',
                        help="list of integers, each indicating the output dimensions of each hidden layer.")
    parser.add_argument('--n_layers', type=int, default=3, help="number of layers.")

    # BasicMatchVAE
    parser.add_argument('--encode_layer_dims_core', type=int, nargs='+', default=None,
                        metavar='N',
                        help="list of integers, each indicating the output dimensions of each hidden layer.")
    parser.add_argument('--encode_layer_dims_branch', type=int, nargs='+', default=None,
                        metavar='N',
                        help="list of integers, each indicating the output dimensions of each hidden layer.")
    parser.add_argument('--decode_x_layer_dims', type=int, nargs='+', default=None, metavar='N',
                        help="list of integers, each indicating the output dimensions of each hidden layer.")
    parser.add_argument('--decode_t_layer_dims', type=int, nargs='+', default=None, metavar='N',
                        help="list of integers, each indicating the output dimensions of each hidden layer.")
    parser.add_argument('--log_propensity_total_weight', type=float, default=2)
    parser.add_argument('--log_propensity_treated_share', type=float, default=0.5)

    # learning rate configs
    parser.add_argument('--init_lr', type=float, default=0.0001, metavar='N',  # 0.002
                        help='initial learning rate for training')
    parser.add_argument('--lr_decay', type=float, default=0.95, metavar='N',
                        help='decay factor of the learning rate every args.update_lr_every_epoch')
    parser.add_argument('--update_lr_every_epoch', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--min_lr', type=float, default=None, metavar='N',  # 0.0002
                        help='the minimum learning rate (lower bound of decaying). If None, no minimum applied.')

    # batch size configs
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for evaluation')
    parser.add_argument('--n_test_batches', type=int, default=-1, metavar='N',
                        help='number of test batches to use per evaluation (-1 uses all available batches)')

    parser.add_argument('--n_epochs', type=int, default=201,
                        metavar='N', help='number of epochs to train')
    parser.add_argument('--seed', type=int, default=1000, metavar='N',
                        help='seed for pseudo-randomness')


    parser.add_argument('--dummy', type=int, default=0, metavar='N')

    # News configs
    parser.add_argument('--news_exp_num', type=int, default=1, metavar='N',
                        help='choose an experiment configuration of News between 1 and 50')
    parser.add_argument('--news_seed', type=int, default=0, metavar='N',
                        help='choose a seed for train/val/test split between 1 and 50')

    # Synthetic1 configs
    parser.add_argument('--synthetic1_seed', type=int, default=1, metavar='N',
                        help='choose an experiment configuration of News between 1 and 50')
    parser.add_argument('--train_size', type=float, default=0.4, metavar='N',
                        help='train size, between 0 and 1 exclusive')
    parser.add_argument('--val_size', type=float, default=0.3, metavar='N',
                        help='train size, between 0 and 1 exclusive')
    parser.add_argument('--n_dims_tot', type=int, default=1000)
    parser.add_argument('--n_samples', type=int, default=2000)


    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.00)
    parser.add_argument('--patience', type=int, default=3)

    # matching config
    parser.add_argument('--aggreg_matching', type=str, default="max")

    # sweep config
    parser.add_argument('--sweep_config', type=str, default=None)
    parser.add_argument('--sweep_id', type=str, default=None)

    # retrieve table
    parser.add_argument('--table_name', type=str, default='results')

    args = parser.parse_args()

    return args


def train(args = None):

    if args is None:
        args = train_args_parser()

    # TODO visualization configs

    import wandb

    # set the right user to login
    with open(args.wandb_config) as f:
        wandb_config = json.load(f)

    wandb.login(key=wandb_config['key'])

    # create W&B logger from with pl support

    # keeping the wandb.init call in, so that everything else keeps working!
    wandb_run = wandb.init(project=wandb_config['project'], entity=wandb_config['entity'], mode=args.wandb_mode, dir=args.wandb_dir)
    wandb_logger = WandbLogger(project=wandb_config['project'], entity=wandb_config['entity'], mode=args.wandb_mode, save_dir=wandb.run.dir)

    # for sweep: don't use such values of args above which are defined by sweep
    # set args value to sweep values instead
    for (key, value) in wandb.config.items():
        setattr(args, key, value)  # args is namespace object

    # update configs -> remember hyperparams
    wandb.config.update(args)

    # sweep configs updating args configs
    # TODO if any

     # load config dictionary instead
    if args.config_args_path != "":
        with open(args.config_args_path, 'rb') as file:
            print("NOTE: Loading args configuration dictionary which overrides any specified hyperparameters!")
            args = pickle.load(file)

    # print out args and wandb run dir
    print(args)
    print("wandb.run.dir: ", wandb.run.dir)

    # assert statements wrt hyperparamters
    # TODO

    # make device a global variable so that dataset.py can access it
    global device
    # initializing global variable (see above)
    device = torch.device(args.device)

    if args.dataset == 'news':
        print(args.data_folder)
        train_data = News(args.news_exp_num, dataset="train", tensor=True, device=args.device, train_size=args.train_size, val_size=args.val_size, data_folder=args.data_folder, seed=args.news_seed)
        val_data = News(args.news_exp_num, dataset="val", tensor=True, device=args.device, train_size=args.train_size, val_size=args.val_size, data_folder=args.data_folder, seed=args.news_seed)
        train_val_data = News(args.news_exp_num, dataset="train", tensor=True, device=args.device, train_size=args.train_size+args.val_size, val_size=0., data_folder=args.data_folder, seed=args.news_seed)
        test_data = News(args.news_exp_num, dataset="test", tensor=True, device=args.device, train_size=args.train_size,val_size=args.val_size, data_folder=args.data_folder, seed=args.news_seed)

        # initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                                                  num_workers=4 if args.device == "cpu" else 0)
    elif args.dataset == 'synthetic1':
        print(args.data_folder)
        train_data = Synthetic1(args.synthetic1_seed, n_samples=args.n_samples, dataset="train", tensor=True, n_dims_tot=args.n_dims_tot, device=args.device, train_size=args.train_size, val_size=args.val_size)
        val_data = Synthetic1(args.synthetic1_seed, n_samples=args.n_samples,dataset="val", tensor=True, n_dims_tot=args.n_dims_tot, device=args.device, train_size=args.train_size, val_size=args.val_size)
        train_val_data = Synthetic1(args.synthetic1_seed, n_samples=args.n_samples,dataset="train", tensor=True, n_dims_tot=args.n_dims_tot, device=args.device, train_size=args.train_size+args.val_size, val_size=0.)
        test_data = Synthetic1(args.synthetic1_seed, n_samples=args.n_samples,dataset="test", tensor=True,  n_dims_tot=args.n_dims_tot, device=args.device, train_size=args.train_size,val_size=args.val_size)

        # initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=4 if args.device == "cpu" else 0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False,
                                                  num_workers=4 if args.device == "cpu" else 0)
    else:
        raise ValueError("args.dataset has not chosen an implemented dataset")

    # dataset-specific parameters
    in_dim = train_data.x.shape[1]
    if args.dataset == 'news':
        likelihoods = 'poisson'
    elif args.dataset == 'synthetic1':
        likelihoods = 'cont'

    # other data-specific hyperparammers
    n_train_batches_per_epoch = len(train_loader)
    model_init_dict = {}
    if args.model_type == 'plain_vae':
        model_init_dict = dict(in_dim=in_dim, z_dim=args.z_dim,
                               encode_layer_dims=args.encode_layer_dims if args.encode_layer_dims is not None else [args.n_hidden] * args.n_layers,
                               decode_layer_dims=args.decode_layer_dims if args.decode_layer_dims is not None else [args.n_hidden] * args.n_layers,
                               likelihoods=likelihoods, dropout_prob=args.dropout_prob, activation=args.activation,
                               dataset=args.dataset, init_lr=args.init_lr)
        wandb.log({'model_init_dict': model_init_dict})
        print('model_init_dict:')
        print(model_init_dict)
        vae = PlainVAE(**model_init_dict)
    elif args.load_model:
        res = torch.load("../outputs/wandb/{}/files/save_dict.pt".format(args.load_model))
        vae = BasicMatchVAE(**res['model_init_dict'])
        vae.load_state_dict(res['state_dict'])
    elif args.model_type == 'basic_match_vae':
        n_layers_t = args.n_layers_t if args.n_layers_t is not None else args.n_layers_core
        n_hidden_t = args.n_hidden_t if args.n_hidden_t is not None else args.n_hidden

        model_init_dict = dict(in_dim=in_dim, z_dim=args.z_dim,
                               encode_layer_dims_core=args.encode_layer_dims_core if args.encode_layer_dims_core is not None else [args.n_hidden] * args.n_layers_core,
                               encode_layer_dims_branch=args.encode_layer_dims_branch if args.encode_layer_dims_branch is not None else [args.n_hidden] * args.n_layers_branch,
                               decode_x_layer_dims=args.decode_x_layer_dims if args.decode_x_layer_dims is not None else [args.n_hidden] * args.n_layers_core,
                               decode_t_layer_dims=args.decode_t_layer_dims if args.decode_t_layer_dims is not None else\
                                   [n_hidden_t + l*args.n_hidden_t_increment for l in range(n_layers_t-1,-1,-1)],
                               likelihoods=likelihoods,  dropout_prob=args.dropout_prob,
                               dataset=args.dataset, init_lr=args.init_lr, activation=args.activation,
                               log_propensity_total_weight=args.log_propensity_total_weight,
                               log_propensity_treated_share=args.log_propensity_treated_share)
        print('model_init_dict:')
        print(model_init_dict)
        wandb.log({'model_init_dict': model_init_dict})
        vae = BasicMatchVAE(**model_init_dict)

    # weights&biases tracking (gradients, network topology)
    wandb.watch(vae)

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(args.seed)

    print("Run trainer...")
    now = datetime.now()
    gpus = -1 if args.device == "cuda" else None
    if args.load_model is None:
        early_stop_callback = EarlyStopping(
            monitor='val_metric/loss',
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode='min'
        )
        trainer = pl.Trainer(deterministic=True, logger=wandb_logger, check_val_every_n_epoch=args.check_val_every_n_epoch, max_epochs=args.n_epochs, gpus=gpus, callbacks=[early_stop_callback])
        # TODO test in the very beginning of training and in the very end!
        if args.do_val_during_training:
            trainer.fit(vae, train_loader, val_loader)
        else:
            trainer.fit(vae, train_loader)
    later = datetime.now()
    diff = later - now
    print("Training took {}".format(diff.seconds))


    # save model
    if args.save_model:
        save_dict_path = os.path.join(wandb.run.dir, "save_dict.pt")
        save_dict = {'state_dict': vae.state_dict(),
                     "model_init_dict": model_init_dict,
                     'args': args}  # args dictionary is already part of saving the model
        torch.save(save_dict, save_dict_path)

    # testing
    now = datetime.now()
    if args.dataset in ['news','synthetic1'] and args.model_type != 'plain_vae':
        if args.device == 'cuda':
            vae = vae.cuda()
        print("TRAINING MATCHING METRICS")
        matching_results, decoder_t_results, scores = matching_metrics(vae, train_val_data,aggreg_matching=args.aggreg_matching)
        for metric in matching_results:
            wandb.log({'train_matching_metric/' + metric: matching_results[metric]})
        for metric in decoder_t_results:
            wandb.log({'train_decoder_t_metric/' + metric: decoder_t_results[metric]})
        save_args_path = os.path.join(wandb.run.dir, "scores_train.pickle")  # args_dict.py
        with open(save_args_path, 'wb') as file:
            pickle.dump(scores, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("TESTING MATCHING METRICS")
        matching_results, decoder_t_results, scores = matching_metrics(vae, test_data, data_to_add=train_val_data, aggreg_matching=args.aggreg_matching)
        for metric in matching_results:
            wandb.log({'test_matching_metric/' + metric: matching_results[metric]})
        for metric in decoder_t_results:
            wandb.log({'test_decoder_t_metric/' + metric: decoder_t_results[metric]})
        save_args_path = os.path.join(wandb.run.dir, "scores_test.pickle")  # args_dict.py
        with open(save_args_path, 'wb') as file:
            pickle.dump(scores, file, protocol=pickle.HIGHEST_PROTOCOL)
    later = datetime.now()
    diff = later - now
    print("Matching took {} seconds".format(diff.seconds))



    # save args config dictionary
    save_args_path = os.path.join(wandb.run.dir, "args.pickle")  # args_dict.py
    with open(save_args_path, 'wb') as file:
        pickle.dump(args, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train()