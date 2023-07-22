import pandas as pd
import numpy as np

import argparse

from plots.utils import get_anomaly_scores, positive_rates
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser
from sklearn.metrics import auc

metric_dict = {
    "auc": auc
}


def run(anomaly_detection_name, model_type, model_names, id_datasets, dataset_names, batch_size, metric_name):

    metric = metric_dict[metric_name]

    model_names = pd.Index(model_names)
    dataset_names = pd.Index(dataset_names)

    df = pd.DataFrame(columns=model_names, index=dataset_names)

    for model_name, id_dataset_name in zip(model_names, id_datasets):

        id_test_anomaly_scores, all_anomaly_scores_list = get_anomaly_scores(anomaly_detection_name, batch_size,
                                                                             id_dataset_name, model_name,
                                                                             dataset_names)

        for ood_anomaly_scores, ood_dataset_name in zip(all_anomaly_scores_list, dataset_names):

            if ood_dataset_name == id_dataset_name:
                continue  # To ensure the scores are genuinely ood

            fpr, tpr = positive_rates(id_test_anomaly_scores, ood_anomaly_scores)

            val = metric(fpr, tpr)
            df[model_name].loc[ood_dataset_name] = val

    print(
        df.to_latex()
    )


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])
parser.add_argument("--metric", choices=metric_dict.keys(),
                    help="The metric by which to measure the success of the anomaly detection method", default="auc")

args = parser.parse_args()
run(args.anomaly_detection, args.model_type, args.model_names, args.id_datasets, args.datasets, args.batch_size, args.metric)
