import torch
import matplotlib.pyplot as plt
import numpy as np

from copy import copy

from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from scipy.stats import kstest

from pandas import DataFrame

from torch.utils.data import DataLoader

from analysis.analysis_utils import get_vanilla_test_dataset


from gradient_serialisation import GRADIENTS_DIR, get_save_file_name


# def norm_dict(grad_dict):
#     return {
#         k: [
#             (grad**2).sum() for grad in grad_dict[k]
#         ]
#         for k in grad_dict.keys()
#     }


# BATCH_SIZE = 10
# MODEL_NAME = "celeba"
# ID_DATASET = "celeba"
# OOD_DATASETS = ["cifar", "svhn"]


def get_single_gradients(batch_size, model_name, id_dataset, ood_datasets):
    id_grads = torch.load(GRADIENTS_DIR +
        get_save_file_name(model_name, id_dataset, batch_size, method="single_grad"))

    ood_grads_list = [
        torch.load(GRADIENTS_DIR +
            get_save_file_name(model_name, ood_dataset, batch_size, method="single_grad"))
        for ood_dataset in ood_datasets
    ]

    return id_grads, ood_grads_list


def get_raw_image_data(batch_size, dataset_name):

    dataset = get_vanilla_test_dataset(dataset_name)
    dl = DataLoader(dataset, batch_size=batch_size)

    return torch.stack([
        x.mean(0).flatten() for x, y in dl
    ])


def get_norms(batch_size, model_name, id_dataset, ood_datasets, printout=False):
    id_norm_file = get_save_file_name(model_name, id_dataset, batch_size)

    ood_norm_files = [
        get_save_file_name(model_name, dataset_name, batch_size) for dataset_name in ood_datasets
    ]

    id_norms = torch.load(GRADIENTS_DIR + id_norm_file)

    ood_norms_list = [
        torch.load(GRADIENTS_DIR + ood_file) for ood_file in ood_norm_files
    ]

    all_norms = copy(ood_norms_list)
    all_norms.append(id_norms)

    all_names = copy(ood_datasets)
    all_names.append(id_dataset)


    zero_keys = set()

    for norms, name in zip(all_norms, all_names):
        zero_count = 0
        for key, value in norms.items():
            zeroes = torch.zeros(len(value))
            if torch.any(value == zeroes):
                zero_keys.add(key)
                zero_count += 1
        if printout:
            print(f"({name}) number of zero gradients: {zero_count}")

    if printout:
        print(f"removing {len(zero_keys)} gradients due to zero values")
    for key in zero_keys:
        for norms in all_norms:
            norms.pop(key)

    return id_norms, ood_norms_list


def log_normalise_gradients(gradient_norms):
    layer_names = gradient_norms[0].keys()
    log_normed_gradient_norms = [
        {} for _ in gradient_norms
    ]

    for layer in layer_names:
        t = torch.cat([
            norm_tensor[layer] for norm_tensor in gradient_norms
        ])

        t = torch.log(t)
        mu = torch.mean(t)
        sigma = torch.std(t)

        for norm_tensor, log_normed in zip(gradient_norms, log_normed_gradient_norms):
            t = torch.log(norm_tensor[layer])
            log_normed[layer] = (t - mu)/sigma

    return log_normed_gradient_norms


# def scatter_plot():
#
#     layer_x_name = "flow.layers.1.actnorm.bias"
#     layer_y_name = "flow.layers.100.actnorm.bias"
#
#     plt.figure(figsize=(10, 10))
#     plt.title(f"Gradient scatter plot (trained {ID_DATASET}, batch size {BATCH_SIZE})")
#
#     for norms, dataset_name in zip([cifar_norms, svhn_norms],
#                                    [f"out of distribution ({OOD_DATASET})", f"in distribution ({ID_DATASET})"]):
#
#         first_layer_norms = norms[layer_x_name][:100]
#         last_layer_norms = norms[layer_y_name][:100]
#
#         plt.scatter(first_layer_norms, last_layer_norms, label=dataset_name)
#
#     plt.legend()
#
#     plt.xlabel(f"log $L^2$ norm ({layer_x_name})")
#     plt.ylabel(f"log $L^2$ norm ({layer_y_name})")
#
#     plt.savefig(f"plots/Gradient scatter plot (trained {ID_DATASET}, batch size {BATCH_SIZE}).png")


def layer_histograms(batch_size, model_name, id_dataset, ood_datasets):
    id_norms, ood_norms_list = get_norms(batch_size, model_name, id_dataset, ood_datasets)
    for n in [1, 10, 40, 80, 100]:
        plt.figure(figsize=(20, 10))
        layer_name = f"flow.layers.{n}.actnorm.bias"

        title = f"Gradient histogram ({model_name}, batch size {batch_size}, {layer_name})"

        plt.title(title)
        plt.xlabel("$\log L^2$ norm")

        log_id_gradients = torch.log(id_norms[layer_name])

        plt.hist(log_id_gradients.numpy(),
                 label=f"in distribution {id_dataset}", density=True, alpha=0.6, bins=40)

        for ood_norms, ood_dataset_name in zip(ood_norms_list, ood_datasets):
            log_ood_gradients = torch.log(ood_norms[layer_name])
            plt.hist(log_ood_gradients.numpy(),
                     label=f"out-of-distribution {ood_dataset_name}", density=True, alpha=0.6, bins=40)

        plt.legend()

        plt.savefig(f"plots/{title}.png")


# def gradient_sum_plot():
#     new_cifar_norms, new_svhn_norms = log_normalise_gradients([cifar_norms, svhn_norms])
#
#     plt.figure(figsize=(20, 10))
#     plt.title(f"Gradient histogram: normalised over the layers")
#     plt.xlabel("log $L^2$ norm")
#
#     for norms, plot_label in zip([new_cifar_norms, new_svhn_norms],
#                                  ["in distribution (cifar)", "out of distribution (svhn)"]):
#         total_norm = sum(
#             norms.values() # [f"flow.layers.{n}.actnorm.bias"] for n in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#         )
#         # total_norm = sum(
#         #     t for t in norms.values() if
#         # )
#
#         plt.hist(total_norm.numpy(),
#                  label=plot_label, density=True, alpha=0.6, bins=40)
#
#     plt.legend()
#     plt.savefig(f"plots/normalised_all_gradients.png")


def get_stacked(norm_dict):
    return torch.stack(
        list(norm_dict.values())
    )


def get_sklearn_norms(id_norms, ood_norms_list):
    """Returns tensors of norms ready for analysis using sklearn."""
    stacked_id_norms = get_stacked(id_norms)
    stacked_id_norms = torch.transpose(stacked_id_norms, 0, 1)

    id_mu = torch.mean(stacked_id_norms, 0)
    id_sigmas = torch.std(stacked_id_norms, 0)

    normed_id_norms = (stacked_id_norms - id_mu)/id_sigmas

    stacked_ood_norms_list = [
        torch.nan_to_num( # Sometimes Nans appear in the input so this replaces them.
        torch.transpose(
            get_stacked(ood_norms),
            0, 1
        )) for ood_norms in ood_norms_list
    ]

    normed_ood_norms_list = [
        (stacked_ood_norms - id_mu) / id_sigmas for stacked_ood_norms in stacked_ood_norms_list
    ]

    return normed_id_norms, normed_ood_norms_list


def split_id_data(id_data, fit_sample_proportion):
    id_samples, layer_count = id_data.shape
    fit_samples = round(id_samples * fit_sample_proportion)

    id_fit = id_data[:fit_samples]
    id_test = id_data[fit_samples:]

    return id_fit, id_test, fit_samples


def fit_logistic_regression_model(logistic_model, id_norms, ood_norms_list, fit_sample_proportion=0.8, **params):
    normed_id_norms, normed_ood_norms_list = get_sklearn_norms(id_norms, ood_norms_list)

    id_fit, id_test, fit_samples = split_id_data(normed_id_norms, fit_sample_proportion)

    # ood_fit = torch.cat([
    #     data[:fit_samples] for data in normed_ood_norms_list
    # ])
    #
    # ood_test = torch.cat([
    #     data[fit_samples:] for data in normed_ood_norms_list
    # ])

    ood_fit = []
    ood_test = []

    for ood_norms in normed_ood_norms_list:
        fit, test, samples = split_id_data(ood_norms, fit_sample_proportion)
        ood_fit.append(fit)
        ood_test.append(test)

    ood_fit = torch.cat(ood_fit)
    ood_test = torch.cat(ood_test)

    X_fit = torch.cat([
        id_fit, ood_fit
    ])

    y_fit = torch.cat([
        torch.zeros(len(id_fit)), torch.ones(len(ood_fit))
    ])

    X_test = torch.cat([
        id_test, ood_test
    ])

    y_test = torch.cat([
        torch.zeros(len(id_test)), torch.ones(len(ood_test))
    ])

    # print(f"Fitting logistic regression with fit samples: {fit_samples}")

    logistic_model.fit(X_fit, y_fit)
    print(f"train score: {logistic_model.score(X_fit, y_fit):1f}")
    print(f"test score: {logistic_model.score(X_test, y_test):1f}")

    print(f"test id rejection: {logistic_model.predict(id_test).mean():1f}")
    print(f"test ood rejection: {logistic_model.predict(ood_test).mean():1f}")

    # print(f"test id predictions: {logistic_model.predict(id_test[:20])}")
    # print(f"test ood predictions: {logistic_model.predict(ood_test[:20])}")


def do_ks_test(id_data, ood_data_list, fit_sample_proportion=0.8):
    id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates = None, None, None

    ood_rejection_rates = []
    for ood_data in ood_data_list:
        ks_stat, p_val = kstest(id_data, ood_data)
        ood_rejection_rates.append(p_val)

    return id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates


def fit_sklearn_unsupervised(normed_id_data, normed_ood_data_list, ModelClass, fit_sample_proportion=0.8, **params):
    def get_rejection_rate(data, model):
        prediction = model.predict(data)
        rejection_rate = (1 - np.mean(prediction))/ 2 # as the prediction is +1 for accept and -1 for reject
        return round(rejection_rate, 3)

    # normed_id_data, normed_ood_data_list = get_sklearn_norms(id_norms, ood_norms_list)

    id_fit, id_test, fit_samples = split_id_data(normed_id_data, fit_sample_proportion)

    model = ModelClass(**params).fit(id_fit)

    id_fit_rejection_rate = get_rejection_rate(id_fit, model)
    id_test_rejection_rate = get_rejection_rate(id_test, model)

    ood_rejection_rates = [
        get_rejection_rate(ood_norms, model) for ood_norms in normed_ood_data_list
    ]

    return id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates


def rejection_rate_table(model_names, dataset_names, get_data, ModelClass, **params):
    rejection_table = {}

    for model_name, id_dataset in zip(model_names, dataset_names):

        id_data, all_data_list = get_data(model_name, id_dataset)

        id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates = fit_sklearn_unsupervised(
            id_data, all_data_list, ModelClass, **params)

        rejection_table[model_name] = [id_fit_rejection_rate, id_test_rejection_rate] + ood_rejection_rates

    table_index = ["fit", "test"] + dataset_names
    return DataFrame(rejection_table, index=table_index)


def norm_rejection_rate_table(batch_size, model_names, dataset_names, ModelClass, **params):

    def get_norm_data(model_name, id_dataset):
        id_norms, all_norms_list = get_norms(batch_size, model_name, id_dataset, dataset_names)

        return get_sklearn_norms(id_norms, all_norms_list)

    return rejection_rate_table(model_names, dataset_names, get_norm_data, ModelClass, **params)


def raw_image_rejection_rate_table(batch_size, dataset_names, ModelClass, **params):
    def get_image_data(model_name, id_dataset):
        all_image_data = []
        for dataset_name in dataset_names:
            raw_image_data = get_raw_image_data(batch_size, dataset_name)
            all_image_data.append(raw_image_data)
            if dataset_name == id_dataset:
                id_image_data = raw_image_data

        return id_image_data, all_image_data

    return rejection_rate_table(dataset_names, dataset_names, get_image_data, ModelClass, **params)


if __name__ == "__main__":

    batch_size = 1
    model_name = "cifar_long"
    id_dataset = "cifar"
    ood_dataset_names = ["svhn", "celeba", "imagenet32"]

    id_norms, ood_norms_list = get_norms(batch_size, model_name, id_dataset, ood_dataset_names)

    for C in [1e-2]:

        layer_names = list(id_norms.keys())

        print(f'fitting logistic regression with: penalty="l1", C={C}, solver="saga"')

        logistic_model = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=1000)
        fit_logistic_regression_model(logistic_model, id_norms, ood_norms_list)

        coeffs = logistic_model.coef_
        coeffs = coeffs.squeeze()
        zero_params = (coeffs == 0).sum()

        print(f"number of params: {coeffs.size}, number of zero params: {zero_params}, non-zero: {coeffs.size - zero_params}")

        # for i, coeff in enumerate(coeffs):
        #     if coeff != 0:
        #         print(i, coeff, layer_names[i])

        print("\n" * 2)

    # model_name = "cifar_long"
    # id_dataset = "cifar"
    # dataset_names = ["svhn", "celeba", "imagenet32"]
    #
    # id_grads, ood_grads_list = get_single_gradients(1, model_name, id_dataset, dataset_names)
    # _, _, p_vals = do_ks_test(id_grads, ood_grads_list)
    #
    # for dataset_name, p_val in zip(dataset_names, p_vals):
    #     print("dataset name:", dataset_name)
    #     print("p value:", p_val)

    # dataset_names = ["FashionMNIST", "MNIST"]
    #
    # model_names = ["PixelCNN_FashionMNIST", "PixelCNN_MNIST"]
    #
    # table_index = ["fit", "test"] + dataset_names
    #
    # print("Rejection tables for gradients:")
    #
    # for batch_size in [32, 10, 5, 1]:
    #     print()
    #     print("batch size", batch_size)
    #     print()
    #
    #     rrt = norm_rejection_rate_table(batch_size, model_names, dataset_names,
    #                                     IsolationForest, fit_sample_proportion=0.6, n_estimators=1000)
    #
    #     print(rrt)
    #
    # print("\n" * 10)
    # print("Rejection table for raw images:")
    #
    # for batch_size in [32, 10, 5, 1]:
    #     print()
    #     print("batch size", batch_size)
    #     print()
    #
    #     rrt = raw_image_rejection_rate_table(batch_size, dataset_names, IsolationForest, fit_sample_proportion=0.6, n_estimators=1000)
    #     print()
    #     print(rrt)



    # fit_logistic_regression_model()
    # fit_sklearn_unsupervised(OneClassSVM, nu=0.1)
    # # fit_sklearn_unsupervised(EllipticEnvelope)
    # fit_sklearn_unsupervised(IsolationForest, fit_sample_proportion=0.6, n_estimators=10000)


