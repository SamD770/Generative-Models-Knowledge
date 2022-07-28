import torch
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

# def norm_dict(grad_dict):
#     return {
#         k: [
#             (grad**2).sum() for grad in grad_dict[k]
#         ]
#         for k in grad_dict.keys()
#     }


cifar_norms = torch.load("cifar_norms.pt")
svhn_norms = torch.load("svhn_norms.pt")

zeroes = torch.zeros(2000)

zero_keys = set()

for norms in cifar_norms, svhn_norms:
    for key, value in norms.items():
        if torch.any(value == zeroes):
            zero_keys.add(key)


print(f"removing {len(zero_keys)} gradients due to zero values")
for key in zero_keys:
    cifar_norms.pop(key)
    svhn_norms.pop(key)


def get_correlations():
    variables = list(cifar_norms.values())
    variables = torch.stack(variables)
    # variables = torch.log(variables)
    return torch.corrcoef(variables)

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
    plt.figure(figsize=(10, 10))
    plt.title(f"Gradient scatter plot")

    for norms, dataset_name in zip([cifar_norms, svhn_norms],
                                   ["in distribution (cifar)", "out of distribution (svhn)"]):


        first_layer_norms = norms["flow.layers.20.actnorm.bias"][:100]
        last_layer_norms = norms["flow.layers.30.actnorm.bias"][:100]

        plt.scatter(first_layer_norms, last_layer_norms, label=dataset_name)

    plt.legend()

    plt.xlabel("log $L^2$ norm (flow.layers.20.actnorm.bias)")
    plt.ylabel("log $L^2$ norm (flow.layers.30.actnorm.bias)")

    plt.savefig("images/gradients_scatter_plot_close.png")


def layer_histograms():
    for n in [1, 10, 40, 80, 100]:
        plt.figure(figsize=(20, 10))
        plt.title(f"Gradient histogram: flow.layers.{n}.actnorm.bias")
        plt.xlabel("$\log$ L^2$ norm")

        log_cifar_gradients = torch.log(cifar_norms[f"flow.layers.{n}.actnorm.bias"])
        log_svhn_gradients = torch.log(svhn_norms[f"flow.layers.{n}.actnorm.bias"])

        plt.hist(log_cifar_gradients.numpy(),
                 label="in distribution (cifar)", density=True, alpha=0.6, bins=40)
        plt.hist(log_svhn_gradients.numpy(),
                 label="out of distribution (svhn)", density=True, alpha=0.6, bins=40)

        plt.legend()

        plt.savefig(f"images/layer_{n}_gradients.png")


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
    plt.savefig(f"images/normalised_all_gradients.png")


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
    # plt.savefig("images/gradients_fitted_Gaussian(2).png")


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


def logistic_regression():
    stacked_cifar_norms = get_stacked(cifar_norms)
    stacked_cifar_norms = torch.transpose(stacked_cifar_norms, 0, 1)

    train_samples = 1500
    test_samples = 2000 - train_samples

    cifar_mu = torch.mean(stacked_cifar_norms, 0)
    cifar_sigmas = torch.std(stacked_cifar_norms, 0)

    normalised_cifar = (stacked_cifar_norms - cifar_mu)/cifar_sigmas
    cifar_train = normalised_cifar[:train_samples]
    cifar_test = normalised_cifar[train_samples:]

    stacked_svhn_norms = get_stacked(svhn_norms)
    stacked_svhn_norms = torch.transpose(stacked_svhn_norms, 0, 1)

    normalised_svhn = (stacked_svhn_norms - cifar_mu)/cifar_sigmas
    svhn_train = normalised_svhn[:train_samples]
    svhn_test = normalised_svhn[train_samples:]

    X_train = torch.cat([
        cifar_train, svhn_train
    ])

    y_train = torch.cat([
        torch.zeros(train_samples), torch.ones(train_samples)
    ])

    X_test = torch.cat([
        cifar_test, svhn_test
    ])

    y_test = torch.cat([
        torch.zeros(test_samples), torch.ones(test_samples)
    ])

    sep_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    print(f"train score: {sep_model.score(X_train, y_train)}")
    print(f"test score: {sep_model.score(X_test, y_test)}")

    print(f"test cifar predictions: {sep_model.predict(cifar_test[:6])}")
    print(f"test svhn predictions: {sep_model.predict(svhn_test[:6])}")


logistic_regression()
