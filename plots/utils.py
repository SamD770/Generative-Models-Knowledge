import sys
from os import path

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from anomaly_methods.utils import anomaly_detection_methods_dict, cache_statistics

from torchvision.utils import make_grid

RUNNING_MODULE_DIR, _ = path.split(sys.argv[0])


# Define parent parsers

def grid_from_imgs(img_seq):
    """Takes a sequence of images and returns a grid with  that can be plotted using plt.imshow"""
    grid = make_grid(img_seq, nrow=8)

    grid = grid.permute(1, 2, 0)

    if grid.min() < 0:  # To account for the fact that the colour datasets are scaled (-0.5, 0.5)
        grid += 0.5

    return grid


def save_plot(title):
    save_filepath = path.join(RUNNING_MODULE_DIR, title + ".png")
    plt.savefig(save_filepath)


def positive_rates(id_test_anomaly_scores, ood_anomaly_scores):
    y_true = torch.cat([torch.ones(len(id_test_anomaly_scores)),
                        torch.zeros(len(ood_anomaly_scores))])
    y_scores = torch.cat([torch.tensor(id_test_anomaly_scores),
                          torch.tensor(ood_anomaly_scores)])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return fpr, tpr


def get_anomaly_scores(anomaly_detection_name, batch_size, id_dataset_name, model_name, all_dataset_names):
    anomaly_detection_method = anomaly_detection_methods_dict[anomaly_detection_name]

    id_dataset_summary, all_dataset_summaries = get_dataset_summmaries(anomaly_detection_method, batch_size,
                                                                       id_dataset_name, model_name, all_dataset_names)

    # Compute anomaly scores
    anomaly_detector = anomaly_detection_method.from_dataset_summary(id_dataset_summary)
    id_fit_summary, id_test_summary = anomaly_detector.split_dataset_summary(id_dataset_summary, 0.5)

    anomaly_detector.setup_method(id_fit_summary)

    id_test_anomaly_scores = anomaly_detector.anomaly_score(id_test_summary)

    all_anomaly_scores_list = [
        anomaly_detector.anomaly_score(dataset_summary) for dataset_summary in all_dataset_summaries
    ]

    return id_test_anomaly_scores, all_anomaly_scores_list


def get_dataset_summmaries(anomaly_detection_method, batch_size, id_dataset_name, model_name, all_dataset_names):

    filepath = anomaly_detection_method.summary_statistic_filepath(
        model_name, id_dataset_name, batch_size
    )

    id_dataset_summary = torch.load(filepath)
    all_dataset_summaries = []

    for dataset_name in all_dataset_names:

        filepath = anomaly_detection_method.summary_statistic_filepath(
            model_name, dataset_name, batch_size)

        all_dataset_summaries.append(
            torch.load(filepath)
        )

    return id_dataset_summary, all_dataset_summaries


def get_dataset_summary(anomaly_detection_method, model_type, model_name, dataset_name, batch_size, create=True):
    filepath = anomaly_detection_method.summary_statistic_filepath(
        model_type, model_name, dataset_name, batch_size)

    if not path.isfile(filepath):
        if create:
            if anomaly_detection_method.model is None:
                pass # load model

            cache_statistics
        else:
            raise FileNotFoundError(f"No statistics cached at {filepath}")


