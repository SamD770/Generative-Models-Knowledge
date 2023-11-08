import os.path
import sys
from os import path

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from anomaly_methods.utils import anomaly_detection_methods_dict
from anomaly_methods.cache_statistics import cache_statistics

from models.utils import load_generative_model

from torchvision.utils import make_grid

RUNNING_MODULE_DIR, _ = path.split(sys.argv[0])


to_styled_dataset_name = {
    "cifar10": "CIFAR-10",
    "svhn": "SVHN",
    "celeba": "CelebA",
    "imagenet32": "ImageNet32",
    "gtsrb": "GTSRB"
}

# Used to enforce a canonical (dataset, colour) pairing to the plots using matplotlib's standard cycle.
dataset_to_colour = {
    dataset_name: colour for dataset_name, colour in zip(
        ["svhn", "celeba", "gtsrb", "cifar10", "imagenet32"],
        plt.rcParams['axes.prop_cycle'].by_key()['color']
    )
}


# Define parent parsers

def grid_from_imgs(img_seq, nrow=8):
    """Takes a sequence of images and returns a grid with  that can be plotted using plt.imshow"""
    grid = make_grid(img_seq, nrow=nrow)

    grid = grid.permute(1, 2, 0)

    if grid.min() < 0:  # To account for the fact that the colour datasets are scaled (-0.5, 0.5)
        grid += 0.5

    return grid


def save_log(title, log_string):
    save_filepath = path.join(RUNNING_MODULE_DIR, title + ".txt")
    with open(save_filepath, "wt") as f:
        f.write(log_string)


def save_plot(title):
    save_filepath = path.join(RUNNING_MODULE_DIR, title + ".png")
    print("saving to:", save_filepath)
    plt.savefig(save_filepath)


def positive_rates(id_test_anomaly_scores, ood_anomaly_scores):
    y_true = torch.cat([torch.ones(len(id_test_anomaly_scores)),
                        torch.zeros(len(ood_anomaly_scores))])
    y_scores = torch.cat([torch.tensor(id_test_anomaly_scores),
                          torch.tensor(ood_anomaly_scores)])
    y_scores = y_scores.nan_to_num()
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return fpr, tpr


def get_anomaly_scores(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset_name, all_dataset_names):

    anomaly_detection_method = anomaly_detection_methods_dict[anomaly_detection_name]

    id_dataset_summary, all_dataset_summaries = get_dataset_summmaries(model_type, model_name, model_mode,
                                                                       anomaly_detection_method, batch_size,
                                                                       id_dataset_name, all_dataset_names)

    # Compute anomaly scores
    anomaly_detector = anomaly_detection_method.from_dataset_summary(id_dataset_summary)
    id_fit_summary, id_test_summary = anomaly_detector.split_dataset_summary(id_dataset_summary, 0.5)

    anomaly_detector.setup_method(id_fit_summary)

    id_test_anomaly_scores = anomaly_detector.anomaly_score(id_test_summary)

    all_anomaly_scores_list = [
        anomaly_detector.anomaly_score(dataset_summary) for dataset_summary in all_dataset_summaries
    ]

    return id_test_anomaly_scores, all_anomaly_scores_list


def get_dataset_summmaries(model_type, model_name, model_mode, anomaly_detection_method, batch_size,
                           id_dataset_name, all_dataset_names):

    id_dataset_summary = load_dataset_summary(model_type, model_name, model_mode, anomaly_detection_method, batch_size,
                                              id_dataset_name)

    all_dataset_summaries = []

    for dataset_name in all_dataset_names:

        dataset_summary = load_dataset_summary(model_type, model_name, model_mode, anomaly_detection_method, batch_size,
                                               dataset_name)

        all_dataset_summaries.append(
            dataset_summary
        )

    return id_dataset_summary, all_dataset_summaries


def load_dataset_summary(model_type, model_name, model_mode, anomaly_detection_method, batch_size, dataset_name, create=True):

    # currently problematic as can lead to model re-loading

    filepath = anomaly_detection_method.summary_statistic_filepath(
        model_type, model_name, model_mode, dataset_name, batch_size
    )

    if not path.isfile(filepath):
        if create:
            print(f"No statistics cached at {filepath},")
            model = load_generative_model(model_type, model_name)
            anomaly_detector = anomaly_detection_method.from_model(model)
            cache_statistics(filepath, anomaly_detector, batch_size, dataset_name, model_mode=model_mode)
        else:

            raise FileNotFoundError(f"No statistics cached at {filepath}, "
                                    f"enable create=True to automatically create them.")

    dataset_summary = torch.load(filepath)

    return dataset_summary

