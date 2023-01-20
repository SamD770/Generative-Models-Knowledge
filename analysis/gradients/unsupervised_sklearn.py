from .gradient_utils import *

import numpy as np

from pandas import DataFrame

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor


def fit_sklearn_unsupervised(normed_id_data, normed_ood_data_list, sklearn_model, fit_sample_proportion=0.8):
    def get_rejection_rate(data, model):
        prediction = model.predict(data)
        rejection_rate = (1 - np.mean(prediction))/ 2 # as the prediction is +1 for accept and -1 for reject
        return round(rejection_rate, 3)

    id_fit, id_test, fit_samples = split_id_data(normed_id_data, fit_sample_proportion)

    sklearn_model.fit(id_fit)

    id_fit_rejection_rate = get_rejection_rate(id_fit, sklearn_model)
    id_test_rejection_rate = get_rejection_rate(id_test, sklearn_model)

    ood_rejection_rates = [
        get_rejection_rate(ood_norms, sklearn_model) for ood_norms in normed_ood_data_list
    ]

    return id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates


def rejection_rate_table(model_names, dataset_names, get_data, ModelClass, **params):
    rejection_table = {}

    for model_name, id_dataset in zip(model_names, dataset_names):

        id_data, all_data_list = get_data(model_name, id_dataset)

        sklearn_model = ModelClass(**params)

        id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates = fit_sklearn_unsupervised(
            id_data, all_data_list, sklearn_model, **params)

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

    dataset_names = ["FashionMNIST", "MNIST"]

    model_names = ["PixelCNN_FashionMNIST", "PixelCNN_MNIST"]

    table_index = ["fit", "test"] + dataset_names

    print("Rejection tables for gradients:")

    for batch_size in [32, 10, 5, 1]:
        print()
        print("batch size", batch_size)
        print()

        rrt = norm_rejection_rate_table(batch_size, model_names, dataset_names,
                                        IsolationForest, fit_sample_proportion=0.6, n_estimators=1000)

        print(rrt)

    print("\n" * 10)
    print("Rejection table for raw images:")

    for batch_size in [32, 10, 5, 1]:
        print()
        print("batch size", batch_size)
        print()

        rrt = raw_image_rejection_rate_table(batch_size, dataset_names, IsolationForest, fit_sample_proportion=0.6, n_estimators=1000)
        print()
        print(rrt)