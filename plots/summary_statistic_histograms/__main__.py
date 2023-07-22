from plots.utils import RUNNING_MODULE_DIR

import torch

import argparse

from anomaly_methods.utils import anomaly_detection_methods_dict
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

import matplotlib.pyplot as plt

from os import path


def run(anomaly_detection_name, model_type, model_name, id_dataset, ood_dataset_names, batch_size):
    anomaly_detection_method = anomaly_detection_methods_dict[anomaly_detection_name]

    filepath = anomaly_detection_method.summary_statistic_filepath(
        model_name, id_dataset, batch_size
    )

    id_dataset_summary = torch.load(filepath)

    ood_dataset_summaries = []

    for dataset_name in ood_dataset_names:
        filepath = anomaly_detection_method.summary_statistic_filepath(
            model_name, dataset_name, batch_size)

        ood_dataset_summaries.append(
            torch.load(filepath)
        )

    anomaly_detector = anomaly_detection_method.from_dataset_summary(id_dataset_summary)

    dir_name = model_name

    for stat_name in anomaly_detector.summary_statistic_names:

        title = f"{stat_name} gradient histogram"

        filepath = path.join(RUNNING_MODULE_DIR, dir_name, title + ".png")

        print(f"creating: {filepath}")
        fig, ax = plt.subplots()
        fig.suptitle(title)

        ax.hist(torch.log(id_dataset_summary[stat_name]).numpy(),
                label=f"in distribution {id_dataset}", density=True, bins=20, alpha=0.6)

        for dataset_name, summary in zip(ood_dataset_names, ood_dataset_summaries):
            ax.hist(torch.log(summary[stat_name]).numpy(),
                    label=f"out-of-distribution {dataset_name}", density=True, bins=20, alpha=0.6)

        fig.legend()

        plt.savefig(filepath)


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])

args = parser.parse_args()
for model_name in args.model_names:
    run(args.anomaly_detection, args.model_type, model_name, args.id_dataset, args.datasets, args.batch_size)
