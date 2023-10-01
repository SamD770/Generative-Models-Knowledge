from plots.utils import save_plot, get_dataset_summmaries, to_styled_dataset_name

import torch
import numpy as np

import argparse

from anomaly_methods.utils import anomaly_detection_methods_dict
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

import matplotlib.pyplot as plt

from os import path

from tqdm import tqdm
import random


def compute_autocorr(index, ordered_statistic_names, summary_statistics, maxlag):

    def get_statistics(name_index):
        statistic_name = ordered_statistic_names[name_index]
        return torch.log(summary_statistics[statistic_name])

    autocorr = np.zeros(maxlag)

    selected_statistic = get_statistics(index)

    for lag in range(maxlag):
        index_other = index + lag
        other_statistic = get_statistics(index_other)

        autocorr[lag] = np.corrcoef(selected_statistic, other_statistic)[0, 1]  # index from correlation matrix

    return autocorr


def compute_autocorr_vector(dataset_summary, max_lag, n_sampled_layers):
    # NOTE: this relies on id_dataset_summary.keys() being ordered.
    # We remove keys with zero standard deviation as this causes zero division errors.
    ordered_statistic_names = [
        key for key in dataset_summary.keys()
        if dataset_summary[key].std() != 0
    ]
    # ordered_statistic_names = list(id_dataset_summary.keys())
    statistic_indices = range(len(ordered_statistic_names) - max_lag)  # We cut off the tail to prevent overshoot
    sampled_statistic_indices = random.choices(statistic_indices, k=n_sampled_layers)
    total_autocorr = np.zeros(max_lag)
    for index in tqdm(sampled_statistic_indices):
        total_autocorr += compute_autocorr(index, ordered_statistic_names, dataset_summary, max_lag)
    mean_autocorr = total_autocorr / n_sampled_layers
    return mean_autocorr


def run(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, datasets, max_lag=400,
        n_sampled_layers=400):

    fontsize = "x-large"

    anomaly_detection_method = anomaly_detection_methods_dict[anomaly_detection_name]

    id_dataset_summary, ood_dataset_summaries = get_dataset_summmaries(model_type, model_name, model_mode,
                                                                       anomaly_detection_method, batch_size,
                                                                       id_dataset, datasets)

    mean_autocorrs = [
        compute_autocorr_vector(dataset_summary, max_lag, n_sampled_layers) for dataset_summary in ood_dataset_summaries
    ]

    fig, ax = plt.subplots(figsize=(10, 7)) # we need a large figure to allow all lines to be plotted

    lags = range(max_lag)
    lags = np.arange(max_lag)

    for mean_autocorr, color, dataset_name, offset in zip(mean_autocorrs, "br", datasets, [0, 0.5]):

        styled_dataset_name = to_styled_dataset_name[dataset_name]
        if dataset_name == id_dataset:
            label = f"in-distribution {styled_dataset_name}"
        else:
            label = f"out-of-distribution {styled_dataset_name}"

        ax.vlines(lags + offset, ymin=0, ymax=mean_autocorr, linewidth=1, colors=color, label=label)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, max_lag)

    ax.set_title(f"autocorrelation of {model_type} layer-wise gradients".title(), fontsize=fontsize)


    ax.set_ylabel("Correlation$(\\log f_{\\mathbf{\\theta}{i}}(\\mathbf{x}), "
                                "\\log f_{\\mathbf{\\theta}{j}}(\\mathbf{x}))$",
                  fontsize=fontsize)
    ax.set_xlabel("$\\vert i - j\\vert$", fontsize=fontsize)

    ax.legend(fontsize=fontsize)

    filename = f"autocorrelation {anomaly_detection_name} {batch_size} {model_type} {model_name}"

    save_plot(filename)


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])

parser.add_argument("--max_lag", type=int, help="the maximum lag at which to cut off the plot")

args = parser.parse_args()
for model_name, id_dataset in zip(args.model_names, args.id_datasets):
    run(args.model_type, model_name, args.model_mode, args.anomaly_detection, args.batch_size, id_dataset, args.datasets,
        args.max_lag)
