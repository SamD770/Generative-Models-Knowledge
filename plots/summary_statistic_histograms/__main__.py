import random

import numpy as np

from plots.utils import RUNNING_MODULE_DIR, get_dataset_summmaries

import torch

import argparse

from anomaly_methods.utils import anomaly_detection_methods_dict
from anomaly_methods.gradients.L2_norms import DistributionFittingL2Norm

from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

import matplotlib.pyplot as plt

from os import path
from os import makedirs


def get_save_dir_path(model_name):
    save_dir_path = path.join(RUNNING_MODULE_DIR, model_name)
    if not path.exists(save_dir_path):
        makedirs(save_dir_path)

    return save_dir_path


def select_summary_stat_names(summary_stat_names, n_statistics):
    if n_statistics is None:
        return summary_stat_names
    else:
        return random.choices(list(summary_stat_names), k=n_statistics)



def run(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names,
        fitted_distribution, n_statistics):

    # Fetch cached statistics from the disk

    anomaly_detection_method = anomaly_detection_methods_dict[anomaly_detection_name]

    id_dataset_summary, ood_dataset_summaries = get_dataset_summmaries(model_type, model_name, model_mode,
                                                                       anomaly_detection_method, batch_size,
                                                                       id_dataset, ood_dataset_names)

    anomaly_detector = anomaly_detection_method.from_dataset_summary(id_dataset_summary)

    if fitted_distribution:
        if not isinstance(anomaly_detector, DistributionFittingL2Norm):
            raise ValueError(f"anomaly detector is of type {type(anomaly_detector)} and thus the fitted "
                            f"distribution(s) can't be plotted.")

        # To fit the distributions, we actually need to run anomaly_detection_method.setup_method

        id_dataset_summary, fit_set_summary = anomaly_detector.split_dataset_summary(id_dataset_summary, 0.5)
        anomaly_detector.setup_method(fit_set_summary)

    save_dir_path = get_save_dir_path(model_name)

    # plot histograms of the data

    selected_stat_names = select_summary_stat_names(anomaly_detector.summary_statistic_names, n_statistics)

    for stat_name in selected_stat_names:

        file_title = f"{stat_name} gradient histogram"

        filepath = path.join(save_dir_path, file_title + ".png")

        print(f"creating: {filepath}")

        figure_title = f"gradients from 1 layer out of {len(anomaly_detector.summary_statistic_names)} in a {model_type} model"
        fig, ax = plt.subplots()
        fig.suptitle(figure_title)

        ax.hist(torch.log(id_dataset_summary[stat_name]).numpy(),
                label=f"in distribution {id_dataset}", density=True, bins=40, alpha=0.6)

        for dataset_name, summary in zip(ood_dataset_names, ood_dataset_summaries):

            if dataset_name == id_dataset:
                continue

            ax.hist(torch.log(summary[stat_name]).numpy(),
                    label=f"out-of-distribution {dataset_name}", density=True, bins=40, alpha=0.6)

        if args.fitted_distribution:
            x_lim = ax.get_xlim()
            x = torch.linspace(*x_lim, steps=100)

            dist = anomaly_detector.fitted_log_scale_distribution(stat_name)
            p = torch.exp(dist.log_prob(x))
            ax.plot(x, p, color="b", label="fitted distribution")

        ax.set_yticks([])
        ax.set_xlabel("$ \\log f_{\\mathbf{\\theta}_\\ell}(\\mathbf{x}_{1:B})  $")

        ax.legend()

        plt.savefig(filepath)


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])

parser.add_argument("--fitted_distribution", action="store_true",
                    help="whether plot the fitted distribution, if it exists, "
                         "with the summary statistics (default False)")

parser.add_argument("--n_statistics", type=int, default=None,                       # We need a string here
                    help="number of statistics to randomly select and plot (defaults to printing all)")

args = parser.parse_args()
for model_name, id_dataset in zip(args.model_names, args.id_datasets):
    run(args.model_type, model_name, args.model_mode, args.anomaly_detection, args.batch_size,
        id_dataset, args.datasets, args.fitted_distribution, args.n_statistics)



