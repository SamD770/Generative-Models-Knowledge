from plots.utils import save_plot, load_dataset_summary

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


def run(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, max_lag=200,
        n_sampled_layers=200):
    anomaly_detection_method = anomaly_detection_methods_dict[anomaly_detection_name]

    id_dataset_summary = load_dataset_summary(model_type, model_name, model_mode,
                                              anomaly_detection_method, batch_size, id_dataset)

    # NOTE: this relies on id_dataset_summary.keys() being ordered.
    # We remove keys with zero standard deviation as this causes zero division errors.
    ordered_statistic_names = [
        key for key in id_dataset_summary.keys()
        if id_dataset_summary[key].std() != 0
    ]
    # ordered_statistic_names = list(id_dataset_summary.keys())
    statistic_indices = range(len(ordered_statistic_names) - max_lag)  # We cut off the tail to prevent overshoot

    sampled_statistic_indices = random.choices(statistic_indices, k=n_sampled_layers)
    total_autocorr = np.zeros(max_lag)

    for index in tqdm(sampled_statistic_indices):
        total_autocorr += compute_autocorr(index, ordered_statistic_names, id_dataset_summary, max_lag)

    mean_autocorr = total_autocorr/n_sampled_layers

    title = f"autocorrelation plot of {anomaly_detection_name} for {model_type} {model_name} evaluated on {id_dataset}"

    fig, ax = plt.subplots()

    lags = range(max_lag)
    ax.vlines(lags, ymin=0, ymax=mean_autocorr, linewidth=0.5)

    ax.set_title(title)

    save_plot(title)


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])

args = parser.parse_args()
for model_name, id_dataset in zip(args.model_names, args.id_datasets):
    run(args.model_type, model_name, args.model_mode, args.anomaly_detection, args.batch_size, id_dataset)
