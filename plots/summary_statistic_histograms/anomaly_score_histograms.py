import matplotlib.pyplot as plt

from plots.summary_statistic_histograms import plot_summary_histograms
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

from plots.utils import get_anomaly_scores

import argparse


def to_summary(anomaly_scores):
    return {
        "anomaly_score": torch.tensor(anomaly_scores)
    }


def run(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names):

    id_test_anomaly_scores, all_anomaly_scores_list = \
        get_anomaly_scores(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names)

    id_anomaly_scores_summary = to_summary(id_test_anomaly_scores)
    all_anomaly_scores_summary = [
        to_summary(anomaly_scores) for anomaly_scores in all_anomaly_scores_list
    ]

    fig, ax = plt.subplots()

    plot_summary_histograms(ax, id_anomaly_scores_summary, id_dataset, all_anomaly_scores_summary, ood_dataset_names, "anomaly_score")

    print("success")


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])
args = parser.parse_args()

for arg_model_name, arg_id_dataset in zip(args.model_names, args.id_datasets):
    run(args.model_type, arg_model_name, args.model_mode, args.anomaly_detection, args.batch_size,
        arg_id_dataset, args.datasets)

