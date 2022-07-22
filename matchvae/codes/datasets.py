
import torch
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from scipy.special import expit
import urllib
import zipfile

class News():

    def __init__(self, exp_num, dataset='train', tensor=True, device="cpu", train_size=0.8, val_size=0.,
                 data_folder=None, seed=0):

        if data_folder is None:
            data_folder = '../data'

        if not os.path.isdir(os.path.join(data_folder, 'News/numpy_dicts/')):
            self._create_data(data_folder)

        with open(os.path.join(data_folder, 'News/numpy_dicts/data_as_dicts_with_numpy_seed_{}'.format(exp_num)),
                  'rb') as file:
            data = pickle.load(file)
        data['cate_true'] = data['mu1'] - data['mu0']

        # Create and store indices
        rng = np.random.default_rng(seed)
        n_rows = len(data['x'])
        original_indices = rng.permutation(n_rows)
        n_train = int(train_size * n_rows)
        if dataset == 'train':
            original_indices = original_indices[:n_train]
        else:
            n_val = int(val_size * n_rows)
            if dataset == 'val':
                if n_val == 0:
                    raise Exception('Validation set empty, please set val_size to a positive float in ]0,1[')
                else:
                    original_indices = original_indices[n_train:n_train + n_val]
            else:
                original_indices = original_indices[n_train + n_val:]  # test set
        self.original_indices = original_indices

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in data.items():
            value = value[original_indices]
            if tensor:
                value = torch.Tensor(value).to(device)
            setattr(self, key, value)

    @staticmethod
    def _create_data(data_folder):

        print('News : no data, creating it')
        print('Downloading zipped csvs')
        urllib.request.urlretrieve('http://www.fredjo.com/files/NEWS_csv.zip', os.path.join(data_folder, 'News/csv.zip'))

        print('Unzipping csvs with sparse data')
        with zipfile.ZipFile(os.path.join(data_folder, 'News/csv.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_folder, 'News'))

        print('Densifying the sparse data')
        os.mkdir(os.path.join(data_folder, 'News/numpy_dicts/'))

        for f_index in range(1, 50 + 1):
            mat = pd.read_csv(os.path.join(data_folder,'News/csv/topic_doc_mean_n5000_k3477_seed_{}.csv.x'.format(f_index)))
            n_rows, n_cols = int(mat.columns[0]), int(mat.columns[1])
            x = np.zeros((n_rows, n_cols)).astype(int)
            for i, j, val in zip(mat.iloc[:, 0], mat.iloc[:, 1], mat.iloc[:, 2]):
                x[i - 1, j - 1] = val
            data = {}
            data['x'] = x
            meta = pd.read_csv(
                os.path.join(data_folder, 'News/csv/topic_doc_mean_n5000_k3477_seed_{}.csv.y'.format(f_index)),
                names=['t', 'y', 'y_cf', 'mu0', 'mu1'])
            for col in ['t', 'y', 'y_cf', 'mu0', 'mu1']:
                data[col] = np.array(meta[col]).reshape((-1, 1))
            with open(os.path.join(data_folder, 'News/numpy_dicts/data_as_dicts_with_numpy_seed_{}'.format(f_index)), 'wb') as file:
                pickle.dump(data, file)

        print('Done!')

    def __getitem__(self, index, attrs=None):
        if attrs is None:
            attrs = ['x', 'y', 't', 'mu0', 'mu1', 'cate_true']
        res = []
        for attr in attrs:
            res.append(getattr(self, attr)[index])
        return (*res,)

    def __len__(self):
        return len(self.original_indices)


class Synthetic1(object):

    def __init__(self, seed=0, n_samples=2000, n_t_dims=10, n_ty_dims=10, n_y_dims=80, scale_t=0.2, intercept_t=0.,
                 scale_noise_y=np.sqrt(0.1), n_dims_tot=1000, dataset='train', tensor=True, device="cpu",
                 train_size=0.8, val_size=0.):
        self.rng = np.random.default_rng(seed=seed)
        x_t = self.rng.normal(size=(n_samples, n_t_dims))
        x_ty = self.rng.normal(size=(n_samples, n_ty_dims))
        x_y = self.rng.normal(size=(n_samples, n_y_dims))
        if n_dims_tot is None:
            n_dims_tot = n_t_dims + n_ty_dims + n_y_dims
        x_other = self.rng.normal(size=(n_samples, n_dims_tot - (n_t_dims + n_ty_dims + n_y_dims)))

        # Covariates
        self.x = np.hstack((x_t, x_ty, x_y, x_other))

        # t
        n_t_dims_tot = n_t_dims + n_ty_dims
        coefs_t = self.rng.normal(size=(1, n_t_dims_tot))
        scalar_prod = np.sum(coefs_t * np.power(np.hstack((x_t, x_ty)), 2), axis=1)
        median_scalar_prod = np.median(scalar_prod)
        self.ps = expit(scale_t * (scalar_prod - median_scalar_prod) + intercept_t).reshape((-1, 1))
        self.t = self.rng.binomial(n=1, p=self.ps)

        # y
        n_y_dims_tot = n_y_dims + n_ty_dims
        coefs_mu0 = self.rng.normal(size=(1, n_y_dims_tot))
        coefs_mu1 = self.rng.normal(size=(1, n_y_dims_tot))
        x_y_all = np.hstack((x_y, x_ty))
        self.mu0 = np.mean(coefs_mu0 * (np.power(x_y_all, 3) + 0.5), axis=1).reshape((-1, 1))
        self.mu1 = np.mean(np.power(x_y_all, 2) + 0.5, axis=1).reshape((-1, 1))
        self.y = self.t * self.mu1 + (1 - self.t) * self.mu0 + self.rng.normal(scale=scale_noise_y,
                                                                               size=(n_samples, 1))

        # Process in train/val/test samples

        data = {
            'x': self.x,
            'y': self.y,
            't': self.t,
            'mu0': self.mu0,
            'mu1': self.mu1,
            'cate_true': self.mu1 - self.mu0
        }

        original_indices = self.rng.permutation(n_samples)
        n_train = int(train_size * n_samples)
        if dataset == 'train':
            original_indices = original_indices[:n_train]
        else:
            n_val = int(val_size * n_samples)
            if dataset == 'val':
                if n_val == 0:
                    raise Exception('Validation set empty, please set val_size to a positive float in ]0,1[')
                else:
                    original_indices = original_indices[n_train:n_train + n_val]
            else:
                original_indices = original_indices[n_train + n_val:]  # test set
        self.original_indices = original_indices

        # Subsample data and convert to torch.Tensor with the right device
        for key, value in data.items():
            value = value[original_indices]
            if tensor:
                value = torch.Tensor(value).to(device)
            setattr(self, key, value)

    def __getitem__(self, index, attrs=None):
        if attrs is None:
            attrs = ['x', 'y', 't', 'mu0', 'mu1', 'cate_true']
        res = []
        for attr in attrs:
            res.append(getattr(self, attr)[index])
        return (*res,)

    def __len__(self):
        return len(self.original_indices)