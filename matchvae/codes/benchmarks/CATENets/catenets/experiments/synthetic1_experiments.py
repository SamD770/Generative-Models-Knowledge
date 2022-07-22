
import os
import numpy as onp
import csv

import sys
print(os.path.abspath("../.."))
sys.path.append("../..")

from datasets import Synthetic1

from sklearn import clone

from catenets.experiments.experiment_utils import eval_root_mse, get_model_set

from catenets.models import PSEUDOOUT_NAME, PseudoOutcomeNet
from catenets.models.transformation_utils import RA_TRANSFORMATION
import numpy as np

# Some constants
DATA_DIR = '../../../data/'
RESULT_DIR = '../../../outputs/outcome_regression_outputs/'
SEP = '_'

# Hyperparameters for experiments on Synthetic1
LAYERS_OUT = 3
LAYERS_R = 3
PENALTY_L2 = 0.01 / 100
PENALTY_ORTHOGONAL = 0

MODEL_PARAMS = {'n_layers_out': LAYERS_OUT, 'n_layers_r': LAYERS_R, 'penalty_l2': PENALTY_L2,
                'penalty_orthogonal': PENALTY_ORTHOGONAL, 'n_layers_out_t': LAYERS_OUT,
                'n_layers_r_t': LAYERS_R, 'penalty_l2_t': PENALTY_L2}

# get basic models
ALL_MODELS = get_model_set(model_selection='all', model_params=MODEL_PARAMS)

COMBINED_MODELS = {PSEUDOOUT_NAME + SEP + RA_TRANSFORMATION + SEP + 'S2':
                            PseudoOutcomeNet(n_layers_r=LAYERS_R, n_layers_out=LAYERS_OUT,
                                             penalty_l2=PENALTY_L2, n_layers_r_t=LAYERS_R,
                                             n_layers_out_t=LAYERS_OUT, penalty_l2_t=PENALTY_L2,
                                             transformation=RA_TRANSFORMATION, first_stage_strategy='S2')}

FULL_MODEL_SET = dict(**ALL_MODELS, **COMBINED_MODELS)


def do_synthetic1_experiments(n_exp: int = 50, file_name: str = 'results_scaled',
                                model_params: dict = None, scale_cate: bool = True,
                                models: dict = None):
    if models is None:
        models = FULL_MODEL_SET
    elif type(models) is list or type(models) is str:
        models = get_model_set(models)

    # make path
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # get file to write in
    out_file = open(RESULT_DIR + file_name + '.csv', 'w', buffering=1)
    writer = csv.writer(out_file)
    header = [name + metric for metric in ['_pehe_in','_pehe_out','_att_true_in','_att_pred_in','_error_att_in','_att_true_out','_att_pred_out','_error_att_out'] for name in models.keys()]
    writer.writerow(header)

    if isinstance(n_exp, int):
        experiment_loop = range(1, n_exp + 1)
    elif isinstance(n_exp, list):
        experiment_loop = n_exp
    else:
        raise ValueError('n_exp should be either an integer or a list of integers.')

    for i_exp in experiment_loop:
        pehe_in = []
        pehe_out = []
        att_true_in = []
        att_pred_in = []
        error_att_in = []
        att_true_out = []
        att_pred_out = []
        error_att_out = []

        # get data
        data_exp = Synthetic1(seed=i_exp, n_samples=2000, n_dims_tot=1000, dataset="train", tensor=False, train_size=0.7, val_size=0.)
        data_exp_test = Synthetic1(seed=i_exp, n_samples=2000, n_dims_tot=1000, dataset="test", tensor=False, train_size=0.7, val_size=0.)

        X, y, w, _, _, cate_true_in = data_exp[:]
        print('X shape :', X.shape)
        print('y shape :', y.shape)
        X_t, _, w_t, _, _, cate_true_out = data_exp_test[:]

        for model_name, estimator in models.items():
            print("Experiment {} with {}".format(i_exp, model_name))
            estimator_temp = clone(estimator)
            if model_params is not None:
                estimator_temp.set_params(**model_params)

            # fit estimator
            estimator_temp.fit(X=X, y=y, w=w)

            cate_pred_in = estimator_temp.predict(X, return_po=False)
            cate_pred_out = estimator_temp.predict(X_t, return_po=False)

            pehe_in.append(eval_root_mse(cate_pred_in, cate_true_in))
            pehe_out.append(eval_root_mse(cate_pred_out, cate_true_out))

            print(cate_true_in[w == 1].mean())
            att_true_in.append(cate_true_in[w == 1].mean())
            att_pred_in.append(cate_pred_in[w == 1].mean())
            error_att_in.append(np.abs(cate_pred_in[w == 1].mean() - cate_true_in[w == 1].mean()))
            print(att_true_in[-1])

            att_true_out.append(cate_true_out[w_t == 1].mean())
            att_pred_out.append(cate_pred_out[w_t == 1].mean())
            error_att_out.append(np.abs(cate_pred_out[w_t == 1].mean() - cate_true_out[w_t == 1].mean()))



        writer.writerow(pehe_in + pehe_out + att_true_in + att_pred_in + error_att_in + att_true_out + att_pred_out + error_att_out)

    out_file.close()
