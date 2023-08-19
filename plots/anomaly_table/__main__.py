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


    metric = metric_dict[metric_name]

    model_names = pd.Index(model_names)
    dataset_names = pd.Index(dataset_names)

    df = pd.DataFrame(columns=model_names, index=dataset_names)

    for model_name, id_dataset_name in zip(model_names, id_datasets):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            id_test_anomaly_scores, all_anomaly_scores_list = get_anomaly_scores(
                model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset_name, dataset_names)

        for ood_anomaly_scores, ood_dataset_name in zip(all_anomaly_scores_list, dataset_names):

            if ood_dataset_name == id_dataset_name:
                continue  # To ensure the scores are genuinely ood

            fpr, tpr = positive_rates(id_test_anomaly_scores, ood_anomaly_scores)

            performance = metric(fpr, tpr)
            df[model_name].loc[ood_dataset_name] = performance

    performance_array = df.to_numpy()
    avg_performance = np.nanmean(performance_array).item()
    stdev_performance = np.nanstd(performance_array).item()
    quantiles = list(np.nanquantile(performance_array, (0.25, 0.50, 0.75)))

    title = f"{metric_name} values for {anomaly_detection_name}, batch size {batch_size} applied to {model_type} " \
            f"in {model_mode} mode, " \
            f"\\newline average performance: {avg_performance:.4f} (stdev: {stdev_performance:.4f})" \
            f"\\newline 25/50/75 quantiles: {quantiles[0]:.4f} / {quantiles[1]:.4f} / {quantiles[2]:.4f}"

    caption = title.replace("_", "\\_")

    def column_name_formatter(column_name):
        if len(column_name) > 18:
            column_name = "\\dots " + column_name[-20:-10]

        return column_name.replace("_", "\\_")

    def row_formatter(row_name):
        styled_name = {
            "cifar10": "CIFAR-10",
            "svhn": "SVHN",
            "celeba": "CelebA",
            "imagenet32": "ImageNet32",
            "gtsrb": "GTSRB"
        }[row_name]

        return "\\texttt{" + styled_name + "}"

    styler = df.style

    styler = styler.format_index(
        formatter=column_name_formatter,
        axis="columns"
    )

    styler = styler.format_index(
        formatter=row_formatter,
        axis="index"
    )

    styler = styler.format(
        na_rep="-",
        precision=4
    )

    table_latex = styler.to_latex(
        hrules=True,
        caption=caption,
        position="H",
        column_format="l | r r r r r"
    )

    # save_log(title, table_latex)

    print(table_latex)


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])
parser.add_argument("--metric", choices=metric_dict.keys(),
                    help="The metric by which to measure the success of the anomaly detection method", default="auc")

args = parser.parse_args()
run(args.model_type, args.model_names, args.model_mode,
    args.anomaly_detection, args.batch_size, args.id_datasets, args.datasets, args.metric)
