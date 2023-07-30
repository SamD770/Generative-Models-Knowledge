from plots.utils import save_plot, load_dataset_summary

import torch
import numpy as np

import argparse

from anomaly_methods.utils import anomaly_detection_methods_dict
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

import matplotlib.pyplot as plt

from os import path

from tqdm import tqdm


def run(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset):
    anomaly_detection_method = anomaly_detection_methods_dict[anomaly_detection_name]

    id_dataset_summary = load_dataset_summary(model_type, model_name, model_mode,
                                              anomaly_detection_method, batch_size, id_dataset)

    # NOTE: this relies on id_dataset_summary.keys() being ordered.

    stacked_dataset_summary = np.stack([
        np.array(val) for val in id_dataset_summary.values()
    ])

    stacked_dataset_summary = stacked_dataset_summary.T

    print(stacked_dataset_summary.shape)

    autocorr_acc = 0.
    n_samples = len(stacked_dataset_summary)

    for X in tqdm(stacked_dataset_summary):
        lags, c, _, _ = plt.acorr(X, maxlags=20)
        autocorr_acc += c

    autocorr_acc = autocorr_acc/n_samples
    print(autocorr_acc.shape)
    print(autocorr_acc)

    title = f"autocorrelation plot of {anomaly_detection_name} for {model_type} {model_name} evaluated on {id_dataset}"

    fig, ax = plt.subplots()

    ax.vlines(lags, ymin=0, ymax=autocorr_acc)

    ax.set_title(title)

    save_plot(title)


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])

args = parser.parse_args()
for model_name, id_dataset in zip(args.model_names, args.id_datasets):
    run(args.model_type, model_name, args.model_mode, args.anomaly_detection, args.batch_size, id_dataset)
