import argparse

from sklearn.metrics import auc, RocCurveDisplay

import matplotlib.pyplot as plt

from plots.utils import save_plot, positive_rates, get_anomaly_scores
from command_line_utils import model_name_parser, anomaly_method_parser, dataset_parser
from anomaly_methods.utils import anomaly_detection_methods_dict


def run(anomaly_detection_name, model_name, id_dataset_name, all_dataset_names, batch_size):

    # Load summaries

    id_test_anomaly_scores, all_anomaly_scores_list = get_anomaly_scores(anomaly_detection_name, batch_size,
                                                                         id_dataset_name, model_name, all_dataset_names)

    # Plot ROC curves

    fig, ax = plt.subplots()

    title = f"ROC plot ({anomaly_detection_name}, {model_name}, Batch size {batch_size})"
    ax.set_title(title)

    for ood_anomaly_scores, ood_dataset_name in zip(all_anomaly_scores_list, all_dataset_names):

        if ood_dataset_name == id_dataset_name:
            continue    # To ensure the scores are genuinely ood

        fpr, tpr = positive_rates(id_test_anomaly_scores, ood_anomaly_scores)

        roc_auc = auc(fpr, tpr)

        print(ood_dataset_name, roc_auc)

        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=ood_dataset_name
        )

        display.plot(ax=ax, label=f"{ood_dataset_name}, AUC={roc_auc:.4f}")

    save_plot(title)


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_name_parser, dataset_parser])

args = parser.parse_args()
for model_name, id_dataset in zip(args.model_names, args.id_datasets):
    run(args.anomaly_detection, model_name, id_dataset, args.datasets, args.batch_size)
