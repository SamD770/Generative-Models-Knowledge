import matplotlib.pyplot as plt
import torch
import numpy as np

from plots.summary_statistic_histograms import plot_summary_histograms
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

from plots.utils import get_anomaly_scores, save_plot

import argparse


def to_summary(anomaly_scores):
    return {
        "anomaly_score": torch.tensor(anomaly_scores)
    }


def run(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names):

    id_test_anomaly_scores, all_anomaly_scores_list = \
        get_anomaly_scores(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names)

    exponents = list(range(1, 6))

    major_ticks = 10**np.array(exponents)
    minor_ticks = np.outer(major_ticks, np.linspace(1, 10, 10))
    minor_ticks = minor_ticks.flatten()

    bois = np.log(major_ticks)
    lil_bois = np.log(minor_ticks)

    lines = list(bois)
    print(lines)

    id_anomaly_scores_summary = to_summary(id_test_anomaly_scores)
    all_anomaly_scores_summary = [
        to_summary(anomaly_scores) for anomaly_scores in all_anomaly_scores_list
    ]

    fig, ax = plt.subplots()

    plot_summary_histograms(ax, id_anomaly_scores_summary, id_dataset,
                            all_anomaly_scores_summary, ood_dataset_names, "anomaly_score")

    exponent_labels = [f"$10^{exponent}$" for exponent in exponents]

    ax.set_xticks(bois, labels=exponent_labels)
    ax.set_xticks(lil_bois, minor=True)
    ax.set_xlabel("Naive $L^2$ norm of $\\nabla_{\\theta} l(\\mathbf{x})$")
    ax.set_xlim(1.5 * np.log(10), 4.8 * np.log(10))

    ax.set_yticks([])

    ax.legend()

    save_plot(f"{model_type} {model_name} {anomaly_detection_name} scores")
    print("success")


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])
args = parser.parse_args()

for arg_model_name, arg_id_dataset in zip(args.model_names, args.id_datasets):
    run(args.model_type, arg_model_name, args.model_mode, args.anomaly_detection, args.batch_size,
        arg_id_dataset, args.datasets)

