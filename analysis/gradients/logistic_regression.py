from gradient_utils import *

from sklearn.linear_model import LogisticRegression


def fit_logistic_regression_model(
    logistic_model, id_norms, ood_norms_list, fit_sample_proportion=0.8, **params
):
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

    X_fit = torch.cat([id_fit, ood_fit])

    y_fit = torch.cat([torch.zeros(len(id_fit)), torch.ones(len(ood_fit))])

    X_test = torch.cat([id_test, ood_test])

    y_test = torch.cat([torch.zeros(len(id_test)), torch.ones(len(ood_test))])

    # print(f"Fitting logistic regression with fit samples: {fit_samples}")

    logistic_model.fit(X_fit, y_fit)
    print(f"train score: {logistic_model.score(X_fit, y_fit):1f}")
    print(f"test score: {logistic_model.score(X_test, y_test):1f}")

    print(f"test id rejection: {logistic_model.predict(id_test).mean():1f}")
    print(f"test ood rejection: {logistic_model.predict(ood_test).mean():1f}")

    # print(f"test id predictions: {logistic_model.predict(id_test[:20])}")
    # print(f"test ood predictions: {logistic_model.predict(ood_test[:20])}")


if __name__ == "__main__":
    batch_size = 1
    model_name = "cifar_long"
    id_dataset = "cifar"
    ood_dataset_names = ["svhn", "celeba", "imagenet32"]

    id_norms, ood_norms_list = get_norms(
        batch_size, model_name, id_dataset, ood_dataset_names
    )

    for C in [1e-2]:
        layer_names = list(id_norms.keys())

        print(f'fitting logistic regression with: penalty="l1", C={C}, solver="saga"')

        logistic_model = LogisticRegression(
            penalty="l1", C=C, solver="saga", max_iter=1000
        )
        fit_logistic_regression_model(logistic_model, id_norms, ood_norms_list)

        coeffs = logistic_model.coef_
        coeffs = coeffs.squeeze()
        zero_params = (coeffs == 0).sum()

        print(
            f"number of params: {coeffs.size}, number of zero params: {zero_params}, non-zero: {coeffs.size - zero_params}"
        )

        # for i, coeff in enumerate(coeffs):
        #     if coeff != 0:
        #         print(i, coeff, layer_names[i])

        print("\n" * 2)
