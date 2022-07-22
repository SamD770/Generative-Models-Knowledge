#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("../..")
from models import PlainVAE
from matching import Matcher

import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.covariance import MinCovDet, EmpiricalCovariance
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn import preprocessing


PROGRAMME_BENCHMARKS_DEFAULT = [
            ('compute_distances', dict(on=['propensity_plain_vae'])),
            #('compute_distances', dict(metric='mahalanobis', control_only='True', on=['x_maha','x_pca_maha'])),
            ('compute_distances', dict(metric='euclidean', on=['plain_vae'])),
            #('trim_ps', dict(eta=0.1, on=['z','balancing','propensity'])),
            #('caliper', dict(value=0.1, on='propensity')),
            ('nearest_neighbor_matching_replacement', dict(n_neighbors=150, on=['plain_vae','propensity_plain_vae'])),
]

METRICS_DEFAULT = ['att','error_att','pehe_treated','linear_mmd_att','rbf_mmd_att','fraction_of_predicted_treated','accuracy','cross-entropy']

aggreg_funcs = {
    'median' : np.median,
    'min' : np.min,
    'max' : np.max
}

from sklearn.linear_model import LogisticRegression

def matching_benchmarks(models, data, data_to_add=None, t_to_add=0, programme=None, metrics=None):
    x, y, t, mu0, mu1, cate_true = data[:]
    print("y shape :", y.shape)
    if data_to_add is not None:
        x_add, y_add, t_add, mu0_add, mu1_add, cate_true_add = data_to_add[:]
        mask = (t_add == t_to_add).ravel()
        print(x_add.shape[0], mask.sum())
        x_add = x_add[mask]
        y_add = y_add[mask]
        mu0_add = mu0_add[mask]
        mu1_add = mu1_add[mask]
        cate_true_add = cate_true_add[mask]
        t_add = t_add[mask]
        x = np.vstack((x, x_add))  # equivalent to torch.vstack, see https://pytorch.org/docs/stable/generated/torch.vstack.html
        y = np.vstack((y, y_add)) 
        t = np.vstack((t, t_add)) 
        mu0 = np.vstack((mu0, mu0_add)) 
        mu1 = np.vstack((mu1, mu1_add)) 
        cate_true = np.vstack((cate_true, cate_true_add)) 
        print("New len x :", x.shape[0])

    if metrics is None:
        metrics = METRICS_DEFAULT

    print("Preparing matchings")
    scores = {}
    model = models['plain_vae']
    model.eval()
    _, q, _ = model(torch.Tensor(x))
    z = q.mean.detach().numpy()
    print(x.shape, z.shape)
    scores["plain_vae"] = z
    # scores["x_maha"] = x
    if "scaler" not in models:
        models["scaler"] = preprocessing.StandardScaler().fit(z)
    if "ps" not in models:
        models["ps"] = LogisticRegression(C=1e6, max_iter=1000).fit(models["scaler"].transform(z), t.ravel())
    scores["propensity_plain_vae"] = models["ps"].predict_proba(z)[:, 1].reshape((-1, 1))

    if programme is None:
        programme = [el for el in PROGRAMME_BENCHMARKS_DEFAULT]
    programme += [
        ('get_treatment_effects', dict(y=y, evaluate=True, ites=cate_true)),
        ('get_balance_metrics', dict(x=x, add_nothing=True)),
    ]

    print([el[0] for el in programme])
    print("Generating matchings")

    m = Matcher(scores, t, att=True, atc=False, propensity_key='propensity_plain_vae')
    results = m.apply_programme(programme)

    balance_df = results[-1]
    te_df = results[-2]

    del scores

    print("Generating results")

    matching_results = {}
    for df in [te_df, balance_df]:
        matching_results.update(
            {
                method + '_' + metric: df.loc[method, metric] \
                for metric in df.columns if metric in metrics \
                for method in df.index
            }
        )

    return matching_results, models

from datasets import News
res = {}
from datetime import datetime
import glob
import os

df_plain_vae_info = pd.read_csv('../../../outputs/sweep_outputs/plainvae_news_infos.csv')
save_dict_filenames = []

for ID, exp_num in zip(df_plain_vae_info.ID, df_plain_vae_info.news_exp_num):
    candidates = [el for el in glob.glob('../../../outputs/wandb/*') if ID in el]
    assert (len(candidates) == 1)
    run_dir = candidates[0]
    save_dict_filename = os.path.join(run_dir, "files/save_dict.pt")
    save_dict_filenames.append((exp_num,save_dict_filename))

for exp_num, save_dict_filename in save_dict_filenames:
    print(exp_num, save_dict_filename)
    now = datetime.now()

    train_data = News(exp_num, dataset="train", tensor=False, train_size=0.7, val_size=0., data_folder="../../../data")
    test_data = News(exp_num, dataset="test", tensor=False, train_size=0.7, val_size=0., data_folder="../../../data")

    save_dict = torch.load(save_dict_filename)
    model = PlainVAE(**save_dict['model_init_dict'])
    model.load_state_dict(save_dict['state_dict'])

    matching_results, models = matching_benchmarks({'plain_vae': model}, train_data)
    for key, value in dict(**matching_results).items():
        res.update({key + '_in': res.get(key + '_in', []) + [value]})
    matching_results, _ = matching_benchmarks(models, test_data, train_data, 0)
    for key, value in matching_results.items():
        res.update({key + '_out': res.get(key + '_out', []) + [value]})
    later = datetime.now()
    diff = later - now
    print("Matching on News {} took {} seconds".format(exp_num, diff.seconds))
        
df_res = pd.DataFrame(res)


df_res.to_csv('../../../outputs/matching_benchmark_outputs/plainvae_news.csv')


