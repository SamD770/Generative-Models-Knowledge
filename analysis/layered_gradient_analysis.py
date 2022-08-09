import torch
import matplotlib.pyplot as plt
import numpy as np

from copy import copy

from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM

from gradient_serialisation import GRADIENTS_DIR, get_save_file_name


# def norm_dict(grad_dict):
#     return {
#         k: [
#             (grad**2).sum() for grad in grad_dict[k]
#         ]
#         for k in grad_dict.keys()
#     }


BATCH_SIZE = 32
MODEL_NAME = "cifar_glow"
ID_DATASET = "cifar"
OOD_DATASETS = ["svhn", "celeba", "imagenet32"]


id_norm_file = get_save_file_name(MODEL_NAME, ID_DATASET, BATCH_SIZE)


ood_norm_files = [
    get_save_file_name(MODEL_NAME, dataset_name, BATCH_SIZE) for dataset_name in OOD_DATASETS
]

id_norms = torch.load(GRADIENTS_DIR + id_norm_file)


ood_norms_list = [
    torch.load(GRADIENTS_DIR + ood_file) for ood_file in ood_norm_files
]

all_norms = copy(ood_norms_list)
all_norms.append(id_norms)

#
# cifar_norm_file = f"svhn_od_norms_{BATCH_SIZE}.pt"
# svhn_norm_file = f"svhn_id_norms_{BATCH_SIZE}.pt"
#
# cifar_norms = torch.load(GRADIENTS_DIR + cifar_norm_file)
# svhn_norms = torch.load(GRADIENTS_DIR + svhn_norm_file)

zero_keys = set()

for norms in all_norms:
    for key, value in norms.items():
        zeroes = torch.zeros(len(value))
        if torch.any(value == zeroes):
            zero_keys.add(key)


print(f"removing {len(zero_keys)} gradients due to zero values")
for key in zero_keys:
    for norms in all_norms:
        norms.pop(key)


# def get_correlations():
#     variables = list(cifar_norms.values())
#     variables = torch.stack(variables)
#     # variables = torch.log(variables)
#     return torch.corrcoef(variables)

#
# correlations = get_correlations()
# print(correlations[:5])


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


def scatter_plot():

    layer_x_name = "flow.layers.1.actnorm.bias"
    layer_y_name = "flow.layers.100.actnorm.bias"

    plt.figure(figsize=(10, 10))
    plt.title(f"Gradient scatter plot (trained {ID_DATASET}, batch size {BATCH_SIZE})")

    for norms, dataset_name in zip([cifar_norms, svhn_norms],
                                   [f"out of distribution ({OOD_DATASET})", f"in distribution ({ID_DATASET})"]):

        first_layer_norms = norms[layer_x_name][:100]
        last_layer_norms = norms[layer_y_name][:100]

        plt.scatter(first_layer_norms, last_layer_norms, label=dataset_name)

    plt.legend()

    plt.xlabel(f"log $L^2$ norm ({layer_x_name})")
    plt.ylabel(f"log $L^2$ norm ({layer_y_name})")

    plt.savefig(f"plots/Gradient scatter plot (trained {ID_DATASET}, batch size {BATCH_SIZE}).png")


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


def gradient_sum_plot():
    new_cifar_norms, new_svhn_norms = log_normalise_gradients([cifar_norms, svhn_norms])

    plt.figure(figsize=(20, 10))
    plt.title(f"Gradient histogram: normalised over the layers")
    plt.xlabel("log $L^2$ norm")

    for norms, plot_label in zip([new_cifar_norms, new_svhn_norms],
                                 ["in distribution (cifar)", "out of distribution (svhn)"]):
        total_norm = sum(
            norms.values() # [f"flow.layers.{n}.actnorm.bias"] for n in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        )
        # total_norm = sum(
        #     t for t in norms.values() if
        # )

        plt.hist(total_norm.numpy(),
                 label=plot_label, density=True, alpha=0.6, bins=40)

    plt.legend()
    plt.savefig(f"plots/normalised_all_gradients.png")


def get_stacked(norm_dict):
    return torch.stack(
        list(norm_dict.values())
    )


def gaussian_fit_plot():
    stacked_cifar_norms = get_stacked(cifar_norms)
    stacked_cifar_norms = stacked_cifar_norms

    train_samples = 1500

    train_cifar_norms = stacked_cifar_norms[:, :train_samples]
    test_cifar_norms = stacked_cifar_norms[:, train_samples:]

    print(f"train cifar shape: {train_cifar_norms.shape}")

    mu = torch.mean(stacked_cifar_norms, 1)
    cov = torch.cov(stacked_cifar_norms)
    L, V = torch.linalg.eig(cov)

    print(L[:180])
    print(f"eigenval sum: {sum(L)}")

    # fitted_gaussian = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    #
    # def get_log_probs(stacked_norms):
    #     transposed_stack = torch.transpose(stacked_norms, 0, 1)
    #     return fitted_gaussian.log_prob(transposed_stack)
    #
    # stacked_svhn_norms = get_stacked(svhn_norms)
    # stacked_svhn_norms = torch.log(stacked_svhn_norms)
    #
    # cifar_log_probs = get_log_probs(test_cifar_norms)
    # svhn_log_probs = get_log_probs(stacked_svhn_norms)
    #
    # print(svhn_log_probs.shape)
    # print(cifar_log_probs.shape)
    #
    # plt.figure(figsize=(20, 10))
    # plt.title(f"Gradient histogram: fitted Gaussian")
    # plt.xlabel("log likelihood")
    #
    # plt.hist(cifar_log_probs.numpy(),
    #          label="in distribution (cifar)", density=True, alpha=0.6, bins=40)
    # plt.hist(svhn_log_probs.numpy(),
    #          label="in distribution (svhn)", density=True, alpha=0.6, bins=40)
    #
    # plt.legend()
    #
    # plt.savefig("plots/gradients_fitted_Gaussian(2).png")


# def principle_components_analysis():
#     stacked_cifar_norms = get_stacked(cifar_norms)
#     stacked_cifar_norms = torch.transpose(stacked_cifar_norms, 0, 1)
#
#     mu = torch.mean(stacked_cifar_norms, 0)
#     sigmas = torch.std(stacked_cifar_norms, 0)
#
#     cifar_bois = (stacked_cifar_norms - mu)/sigmas
#
#
#     covariance = torch.cov(
#         torch.transpose(cifar_bois, 0, 1)
#     )
#     print(torch.mean(cifar_bois))
#     print(torch.std(cifar_bois, 0))
#
#     evals, evecs = torch.linalg.eig(covariance)
#     evals = torch.real(evals)
#     evecs = torch.real(evecs)
#
#     print(evecs.shape)
#
#     evec_cutoff = 50
#     principle_evecs = evecs[:evec_cutoff]


def fit_gradient_models():
    stacked_id_norms = get_stacked(id_norms)
    stacked_id_norms = torch.transpose(stacked_id_norms, 0, 1)

    id_samples, layer_count = stacked_id_norms.shape

    fit_samples = 600
    test_samples = id_samples - fit_samples

    print(f"total samples: {id_samples} fit samples: {fit_samples} layer count: {layer_count}")

    id_mu = torch.mean(stacked_id_norms, 0)
    id_sigmas = torch.std(stacked_id_norms, 0)

    normed_id_norms = (stacked_id_norms - id_mu)/id_sigmas
    id_fit = normed_id_norms[:fit_samples]
    id_test = normed_id_norms[fit_samples:]

    stacked_ood_norms_list = [
        torch.transpose(
            get_stacked(ood_norms),
            0, 1
        ) for ood_norms in ood_norms_list
    ]

    ood_samples_list = [
        stacked_ood_norms.shape[0] for stacked_ood_norms in stacked_ood_norms_list
    ]

    normed_ood_norms_list = [
        (stacked_ood_norms - id_mu) / id_sigmas for stacked_ood_norms in stacked_ood_norms_list
    ]

    # ood_fit = normed_ood_norms[:fit_samples]
    # ood_test = normed_ood_norms[fit_samples:]
    #
    # X_train = torch.cat([
    #     id_fit, ood_fit
    # ])
    #
    # y_train = torch.cat([
    #     torch.zeros(fit_samples), torch.ones(fit_samples)
    # ])
    #
    # X_test = torch.cat([
    #     id_test, ood_test
    # ])
    #
    # y_test = torch.cat([
    #     torch.zeros(test_samples), torch.ones(ood_samples - fit_samples)
    # ])

    # logistic_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    #
    # print(f"train id score: {logistic_model.score(X_train, y_train)}")
    # print(f"test ood score: {logistic_model.score(X_test, y_test)}")
    #
    # print(f"test id predictions: {logistic_model.predict(id_test[:10])}")
    # print(f"test ood predictions: {logistic_model.predict(ood_test[:10])}")

    print()
    print("support vector machine results")

    svm_model = OneClassSVM().fit(id_fit)
    in_dist_train_prediction = svm_model.predict(id_fit)
    in_dist_test_prediction = svm_model.predict(id_test)

    print(f"svm train prediction for {ID_DATASET}: {in_dist_train_prediction[:10]} mean: {np.mean(in_dist_train_prediction)}")
    print(f"svm test prediction for {ID_DATASET}:{in_dist_train_prediction[:10]} mean: {np.mean(in_dist_test_prediction)}")

    for normed_ood_norms, ood_dataset_name in zip(normed_ood_norms_list, OOD_DATASETS):
        nan_count = np.count_nonzero(np.isnan(normed_ood_norms))
        non_nan_count = np.count_nonzero(~np.isnan(normed_ood_norms))
        print(f"{ood_dataset_name}  Clean count: {non_nan_count}  NaN count: {nan_count}")

        out_dist_prediction = svm_model.predict(normed_ood_norms)
        print(f"svm ood prediction for {ood_dataset_name}: {out_dist_prediction[:10]} mean: {np.mean(out_dist_prediction)}")

    # print(type(svm_model.predict(svhn_test)))
    # print(type(svm_model.predict(id_test)))
    #
    # print("svm fit done")
    #
    # return svm_model, ood_test, id_test


    # print(f"model coeffs: {logistic_model.coef_}")
    #
    # model_coeffs_tensor = torch.tensor(logistic_model.coef_)
    # model_coeffs_tensor = torch.squeeze(model_coeffs_tensor)

    # plt.figure(figsize=(20, 10))
    # plt.title(f"Gradient histogram: normalised over the layers using weighting from logistic regression (no log)")
    # plt.xlabel("weighted $L^2$ norm")
    #
    # for normalised_norms, label in zip([normed_id_norms, normed_ood_norms],
    #                                    ["cifar (in distribution)", "svhn (ood)"]):
    #     scores = torch.sum(normalised_norms*model_coeffs_tensor, 1)
    #     log_scores = torch.log(scores)
    #
    #     plt.hist(scores.numpy(),
    #              label=label, density=True, alpha=0.6, bins=40)
    #
    # plt.legend()
    # plt.savefig("plots/normalised_using_logistic_regression.png")


if __name__ == "__main__":
    layer_histograms()
