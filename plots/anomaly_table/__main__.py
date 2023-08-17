import pandas as pd
import numpy as np

import argparse
import warnings

from plots.utils import get_anomaly_scores, positive_rates, save_log
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser
from sklearn.metrics import auc

metric_dict = {
    "auc": auc
}


def run(model_type, model_names, model_mode, anomaly_detection_name, batch_size, id_datasets, dataset_names, metric_name):

    performances = []

    metric = metric_dict[metric_name]

    model_names = pd.Index(model_names)
    dataset_names = pd.Index(dataset_names)

    df = pd.DataFrame(columns=model_names, index=dataset_names)

    for model_name, id_dataset_name in zip(model_names, id_datasets):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            id_test_anomaly_scores, all_anomaly_scores_list = get_anomaly_scores(model_type, model_name, model_mode,
                                                                             anomaly_detection_name, batch_size,
                                                                             id_dataset_name, dataset_names)

        for ood_anomaly_scores, ood_dataset_name in zip(all_anomaly_scores_list, dataset_names):

            if ood_dataset_name == id_dataset_name:
                continue  # To ensure the scores are genuinely ood

            fpr, tpr = positive_rates(id_test_anomaly_scores, ood_anomaly_scores)

            performance = metric(fpr, tpr)
            df[model_name].loc[ood_dataset_name] = performance

    title = f"{metric_name} values for {anomaly_detection_name}, batch size {batch_size} applied to {model_type} " \
            f"in {model_mode} mode"

    def column_formatter(column_name):
        if len(column_name) > 18:
            column_name = "\\dots " + column_name[-16:]

        return column_name.replace("_", "\\_")

    styler = df.style.format_index(
        formatter=column_formatter,
        axis="columns"
    )

    caption = title.replace("_", "\\_")

    table_latex = styler.to_latex(
        hrules=True,
        caption=caption,
        position="h!"
    )

    save_log(title, table_latex)

    print(table_latex)

    avg_performance = np.nanmean(df.to_numpy()).item()
    print(f"average performance: {avg_performance:.4f}")
    print()
    print()


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])
parser.add_argument("--metric", choices=metric_dict.keys(),
                    help="The metric by which to measure the success of the anomaly detection method", default="auc")

args = parser.parse_args()
run(args.model_type, args.model_names, args.model_mode,
    args.anomaly_detection, args.batch_size, args.id_datasets, args.datasets, args.metric)
