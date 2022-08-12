import torch
import matplotlib.pyplot as plt
import numpy as np

from copy import copy

from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from pandas import DataFrame


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


def get_norms(batch_size, model_name, id_dataset, ood_datasets):
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
        print(f"({name}) number of zero gradients: {zero_count}")


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

        # print(layer)
        # print(f"mu {mu}")
        # print(f"sigma {sigma}")

        for norm_tensor, log_normed in zip(gradient_norms, log_normed_gradient_norms):
            t = torch.log(norm_tensor[layer])
            log_normed[layer] = (t - mu)/sigma
            # print(f"individual mu: {torch.mean(t)}")
            # print(f"individual sigma: {torch.std(t)}")

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


def layer_histograms():
    for n in [1, 10, 40, 80, 100]:
        plt.figure(figsize=(20, 10))
        layer_name = f"flow.layers.{n}.actnorm.bias"

        title = f"Gradient histogram ({MODEL_NAME}, batch size {BATCH_SIZE}, {layer_name})"

        plt.title(title)
        plt.xlabel("$\log L^2$ norm")

        log_id_gradients = torch.log(id_norms[layer_name])

        plt.hist(log_id_gradients.numpy(),
                 label=f"in distribution {ID_DATASET}", density=True, alpha=0.6, bins=40)

        for ood_norms, ood_dataset_name in zip(ood_norms_list, OOD_DATASETS):
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


# def gaussian_fit_plot():
#     stacked_cifar_norms = get_stacked(cifar_norms)
#     stacked_cifar_norms = stacked_cifar_norms
#
#     train_samples = 1500
#
#     train_cifar_norms = stacked_cifar_norms[:, :train_samples]
#     test_cifar_norms = stacked_cifar_norms[:, train_samples:]
#
#     print(f"train cifar shape: {train_cifar_norms.shape}")
#
#     mu = torch.mean(stacked_cifar_norms, 1)
#     cov = torch.cov(stacked_cifar_norms)
#     L, V = torch.linalg.eig(cov)
#
#     print(L[:180])
#     print(f"eigenval sum: {sum(L)}")
#
#     # fitted_gaussian = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
#     #
#     # def get_log_probs(stacked_norms):
#     #     transposed_stack = torch.transpose(stacked_norms, 0, 1)
#     #     return fitted_gaussian.log_prob(transposed_stack)
#     #
#     # stacked_svhn_norms = get_stacked(svhn_norms)
#     # stacked_svhn_norms = torch.log(stacked_svhn_norms)
#     #
#     # cifar_log_probs = get_log_probs(test_cifar_norms)
#     # svhn_log_probs = get_log_probs(stacked_svhn_norms)
#     #
#     # print(svhn_log_probs.shape)
#     # print(cifar_log_probs.shape)
#     #
#     # plt.figure(figsize=(20, 10))
#     # plt.title(f"Gradient histogram: fitted Gaussian")
#     # plt.xlabel("log likelihood")
#     #
#     # plt.hist(cifar_log_probs.numpy(),
#     #          label="in distribution (cifar)", density=True, alpha=0.6, bins=40)
#     # plt.hist(svhn_log_probs.numpy(),
#     #          label="in distribution (svhn)", density=True, alpha=0.6, bins=40)
#     #
#     # plt.legend()
#     #
#     # plt.savefig("plots/gradients_fitted_Gaussian(2).png")


def get_sklearn_norms(id_norms, ood_norms_list):
    """Returns tensors of norms ready for analysis using sklearn."""
    stacked_id_norms = get_stacked(id_norms)
    stacked_id_norms = torch.transpose(stacked_id_norms, 0, 1)

    # id_samples, layer_count = stacked_id_norms.shape
    # test_samples = id_samples - fit_samples

    # print(f"total samples: {id_samples} fit samples: {fit_samples} layer count: {layer_count}")

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
    #
    # ood_samples_list = [
    #     stacked_ood_norms.shape[0] for stacked_ood_norms in stacked_ood_norms_list
    # ]

    return normed_id_norms, normed_ood_norms_list


def split_id_data(id_data, fit_sample_proportion):
    id_samples, layer_count = id_data.shape
    fit_samples = round(id_samples * fit_sample_proportion)

    id_fit = id_data[:fit_samples]
    id_test = id_data[fit_samples:]

    return id_fit, id_test, fit_samples


def fit_logistic_regression_model(id_norms, ood_norms_list, fit_sample_proportion=0.8):
    normed_id_norms, normed_ood_norms_list = get_sklearn_norms(id_norms, ood_norms_list)

    id_fit, id_test, fit_samples = split_id_data(normed_id_norms, fit_sample_proportion)

    ood_fit = torch.cat([
        data[:fit_samples] for data in normed_ood_norms_list
    ])

    ood_test = torch.cat([
        data[fit_samples:] for data in normed_ood_norms_list
    ])

    X_fit = torch.cat([
        id_fit, ood_fit
    ])

    y_fit = torch.cat([
        torch.ones(len(id_fit)), torch.zeros(len(ood_fit))
    ])

    X_test = torch.cat([
        id_test, ood_test
    ])

    y_test = torch.cat([
        torch.ones(len(id_test)), torch.zeros(len(ood_test))
    ])

    print(f"Fitting logistic regression with fit samples: {fit_samples}")

    logistic_model = LogisticRegression(max_iter=1000).fit(X_fit, y_fit)

    print(f"train score: {logistic_model.score(X_fit, y_fit)}")
    print(f"test score: {logistic_model.score(X_test, y_test)}")

    print(f"test id predictions: {logistic_model.predict(id_test[:20])}")
    print(f"test ood predictions: {logistic_model.predict(ood_test[:20])}")


def fit_sklearn_unsupervised(id_norms, ood_norms_list, ModelClass, fit_sample_proportion=0.8, **params):
    def get_rejection_rate(data, model):
        prediction = model.predict(data)
        rejection_rate = (1 - np.mean(prediction))/ 2 # as the prediction is +1 for accept and -1 for reject
        return round(rejection_rate, 3)

    normed_id_norms, normed_ood_norms_list = get_sklearn_norms(id_norms, ood_norms_list)

    id_fit, id_test, fit_samples = split_id_data(normed_id_norms, fit_sample_proportion)

    # print()
    # print(f"fitting {ModelClass} with {params} and fit samples {fit_samples}")

    model = ModelClass(**params).fit(id_fit)

    id_fit_rejection_rate = get_rejection_rate(id_fit, model)
    id_test_rejection_rate = get_rejection_rate(id_test, model)

    ood_rejection_rates = [
        get_rejection_rate(ood_norms, model) for ood_norms in normed_ood_norms_list
    ]

    return id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates


    # print()
    # print(f"rejection rate for in-distribution {ID_DATASET} fit: {get_rejection_rate(id_fit, model)}")
    # print(f"rejection rate for in-distribution {ID_DATASET} test: {get_rejection_rate(id_test, model)}")
    #
    # for normed_ood_norms, ood_dataset_name in zip(normed_ood_norms_list, OOD_DATASETS):
    #
    #     print(f"rejection rate for ood {ood_dataset_name}: {get_rejection_rate(normed_ood_norms, model)}")
    # print()
    # print()


def rejection_rate_table(batch_size, model_names, dataset_names, ModelClass, **params):
    rejection_table = {}

    for model_name, id_dataset in zip(model_names, dataset_names):

        print("getting rejection rates for:", model_name, id_dataset)

        id_norms, all_norms_list = get_norms(batch_size, model_name, id_dataset, dataset_names)

        id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates = fit_sklearn_unsupervised(
            id_norms, all_norms_list, ModelClass, **params)

        rejection_table[model_name] = [id_fit_rejection_rate, id_test_rejection_rate] + ood_rejection_rates

    table_index = ["fit", "test"] + dataset_names
    return DataFrame(rejection_table, index=table_index)


if __name__ == "__main__":

    model_names = ["cifar_long", "svhn_working", "celeba", "imagenet32"]
    dataset_names = ["cifar", "svhn", "celeba", "imagenet32"]

    table_index = ["fit", "test"] + dataset_names

    for batch_size in [32]:
        print()
        print("batch size", batch_size)
        print()

        print(rejection_rate_table(batch_size, model_names, dataset_names,
                                   IsolationForest, fit_sample_proportion=0.6, n_estimators=10000))

    # fit_logistic_regression_model()
    # fit_sklearn_unsupervised(OneClassSVM, nu=0.1)
    # # fit_sklearn_unsupervised(EllipticEnvelope)
    # fit_sklearn_unsupervised(IsolationForest, fit_sample_proportion=0.6, n_estimators=10000)


