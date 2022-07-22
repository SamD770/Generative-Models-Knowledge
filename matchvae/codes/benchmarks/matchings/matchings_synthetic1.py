#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import sys
sys.path.append("../..")
from models import BasicMatchVAE
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
            ('compute_distances', dict(on=['propensity_x','propensity_x_pca'])),
            ('compute_distances', dict(metric='euclidean', on=['x_eucl','x_pca_eucl'])),
            ('nearest_neighbor_matching_replacement', dict(n_neighbors=150, on=['x_eucl','propensity_x','x_pca_eucl','propensity_x_pca'])), #'x_maha','x_pca_maha',
            ('random_matching_replacement', dict(n_neighbors=150, on=['random'])), #'x_maha','x_pca_maha',
            ('ipw', dict(on=['ipw_ps_x','ipw_ps_x_pca'], eps=1e-3)) #
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
    scores["x_eucl"] = x
    scores["random"] = x
    #scores["x_maha"] = x
    if "scaler" not in models:
        models["scaler"] = preprocessing.StandardScaler().fit(x)
    if "ps" not in models:
        models["ps"] = LogisticRegression(C=1e6, max_iter=1000).fit(models["scaler"].transform(x), t.ravel())
    scores["propensity_x"] = models["ps"].predict_proba(x)[:, 1].reshape((-1,1))
    scores['ipw_ps_x'] = scores["propensity_x"]
    if "pca" not in models:
        models["pca"] = PCA(n_components=20).fit(x)
    scores["x_pca_eucl"] = models["pca"].transform(x)
    print("PCA dims", scores["x_pca_eucl"].shape)
    #scores["x_pca_maha"] = scores["x_pca_eucl"]
    if "scaler_pca" not in models:
        models["scaler_pca"] = preprocessing.StandardScaler().fit(scores["x_pca_eucl"])
    if "ps_pca" not in models:
        models["ps_pca"] = LogisticRegression(C=1e6, max_iter=1000).fit(models["scaler_pca"].transform(scores["x_pca_eucl"]), t.ravel())
    scores["propensity_x_pca"] = models["ps_pca"].predict_proba(scores["x_pca_eucl"])[:, 1].reshape((-1,1))
    scores['ipw_ps_x_pca'] = scores["propensity_x_pca"]
    

    if programme is None:
        programme = [el for el in PROGRAMME_BENCHMARKS_DEFAULT]
    programme += [
        ('get_treatment_effects', dict(y=y, evaluate=True, ites=cate_true)),
        ('get_balance_metrics', dict(x=x, add_nothing=True)),
    ]

    print([el[0] for el in programme])
    print("Generating matchings")

    m = Matcher(scores, t, att=True, atc=False, propensity_key='propensity_x')
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

from datasets import Synthetic1
res = {}
from datetime import datetime

for exp_num in range(1,50+1):
    now = datetime.now()

    train_data = Synthetic1(exp_num, n_samples=2000, n_dims_tot=1000, dataset="train", tensor=False, train_size=0.7, val_size=0.)
    test_data = Synthetic1(exp_num, n_samples=2000, n_dims_tot=1000, dataset="test", tensor=False, train_size=0.7, val_size=0.)
    
    matching_results, models = matching_benchmarks({}, train_data)
    for key,value in dict(**matching_results).items():
        res.update({key+'_in': res.get(key+'_in',[]) + [value]})
    matching_results, _ = matching_benchmarks(models, test_data, train_data, 0)
    for key,value in matching_results.items():
        res.update({key+'_out': res.get(key+'_out',[]) + [value]})
    later = datetime.now()
    diff = later - now
    print("Matching on Synthetic1 {} took {} seconds".format(exp_num, diff.seconds))
        
df_res = pd.DataFrame(res)


df_res.to_csv('../../../outputs/matching_benchmark_outputs/synthetic1.csv')


