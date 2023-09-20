import numpy as np

from plots.utils import get_anomaly_scores, positive_rates

import pandas as pd
import warnings

from plots.utils import get_anomaly_scores, positive_rates, save_log

from sklearn.metrics import auc


def get_dataframe(anomaly_detection_name, batch_size, dataset_names, id_datasets, metric_name, model_mode, model_names,
                  model_name_column, model_type):

    metric = metric_dict[metric_name]
    dataset_names = pd.Index(dataset_names)
    if model_name_column:
        column_names = pd.Index(model_names)
    else:
        column_names = pd.Index(id_datasets)
    df = pd.DataFrame(columns=column_names, index=dataset_names)
    for model_name, id_dataset_name in zip(model_names, id_datasets):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            id_test_anomaly_scores, all_anomaly_scores_list = get_anomaly_scores(
                model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset_name, dataset_names)

        for ood_anomaly_scores, ood_dataset_name in zip(all_anomaly_scores_list, dataset_names):

            if ood_dataset_name == id_dataset_name:
                continue  # To ensure the scores are genuinely ood

            fpr, tpr = positive_rates(id_test_anomaly_scores, ood_anomaly_scores)

            performance = metric(fpr, tpr)

            if model_name_column:
                key = model_name
            else:
                key = id_dataset_name

            df[key].loc[ood_dataset_name] = performance
    return df


def model_name_formatter(column_name):
    if len(column_name) > 18:
        column_name = "\\dots " + column_name[-20:-10]

    return column_name.replace("_", "\\_")


def dataset_name_formatter(row_name):
    styled_name = {
        "cifar10": "CIFAR-10",
        "svhn": "SVHN",
        "celeba": "CelebA",
        "imagenet32": "ImageNet32",
        "gtsrb": "GTSRB"
    }[row_name]

    return "\\texttt{" + styled_name + "}"


metric_dict = {
    "auc": auc
}


def get_performance_stats(df):
    performance_array = df.to_numpy()
    avg_performance = np.nanmean(performance_array).item()
    stdev_performance = np.nanstd(performance_array).item()
    quantiles = list(np.nanquantile(performance_array, (0.25, 0.50, 0.75)))
    return avg_performance, quantiles, stdev_performance
