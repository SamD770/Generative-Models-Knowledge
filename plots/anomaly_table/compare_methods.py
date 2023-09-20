from plots.anomaly_table import get_dataframe, metric_dict, model_name_formatter, dataset_name_formatter, get_performance_stats
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

from anomaly_methods.utils import anomaly_detection_methods

import argparse
import pandas as pd


def anomaly_method_name_formatter(anomaly_detection_name):
    anomaly_detection_name = anomaly_detection_name.replace("_", "\\_")

    # We rotate the anomaly detection name in the table
    return "\\rotatebox[origin=c]{90}{" + anomaly_detection_name + "}"


def run(method_1, method_2, model_type, model_names, model_mode, batch_size, id_datasets, dataset_names,
        metric_name, model_name_column):

    df_method_1 = get_dataframe(method_1, batch_size, dataset_names,
                                id_datasets, metric_name, model_mode, model_names, model_name_column, model_type)

    df_method_2 = get_dataframe(method_2, batch_size, dataset_names,
                                id_datasets, metric_name, model_mode, model_names, model_name_column, model_type)

    df_comparison = df_method_2 > df_method_1

    # A silly little hack to store the float values which we cant to be bold

    bold_flag_dict = {
        performance: greater_than for performance, greater_than in zip(df_method_2.stack(dropna=False), df_comparison.stack())
    } | {
        performance: not greater_than for performance, greater_than in zip(df_method_1.stack(dropna=False), df_comparison.stack())
    }

    def bold_formatter(performance):

        if pd.isna(performance):
            return "-"

        bold_flag = bold_flag_dict[performance]

        performance = format(performance, ".4f")

        if bold_flag:
            performance = "\\textbf{" + performance + "}"

        return performance

    df = pd.concat([df_method_1, df_method_2],
                   keys=[method_1, method_2])

    styler = df.style

    if model_name_column:
        column_formatter = model_name_formatter
    else:
        column_formatter = dataset_name_formatter

    styler = styler.format_index(
        formatter=column_formatter,
        axis="columns"
    )

    styler = styler.format_index(
        formatter=anomaly_method_name_formatter,
        axis="index",
        level=0
    )

    styler = styler.format_index(
        formatter=dataset_name_formatter,
        axis="index",
        level=1
    )

    styler = styler.format(
        formatter=bold_formatter
    )

    avg_performance_1, quantiles_1, _ = get_performance_stats(df_method_1)
    avg_performance_2, quantiles_2, _ = get_performance_stats(df_method_2)

    caption = f"{metric_name} values for {method_1} [top] and {method_2} [bottom], batch size {batch_size} applied to {model_type} " \
            f"\\newline average performance for {method_1}: {avg_performance_1:.4f}, \\quad 25/50/75 quantiles: {quantiles_1[0]:.4f} / {quantiles_1[1]:.4f} / {quantiles_1[2]:.4f}" \
            f"\\newline average performance for {method_2}: {avg_performance_2:.4f}, \\quad 25/50/75 quantiles: {quantiles_2[0]:.4f} / {quantiles_2[1]:.4f} / {quantiles_2[2]:.4f}"

    caption = caption.replace("_", "\\_")

    table_latex = styler.to_latex(
        hrules=True,
        position="H",
        caption=caption,
        clines="skip-last;data",
        column_format="l | l | r r r r r"
    )

    print(table_latex)

    df_comparison = (df_method_2 > df_method_1)


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])

parser.add_argument("--compare_to", choices=anomaly_detection_methods,
                    help="the anomaly detection method to compare to (bottom of table)")
parser.add_argument("--metric", choices=metric_dict.keys(),
                    help="The metric by which to measure the success of the anomaly detection method", default="auc")
parser.add_argument("--model_name_column", action="store_true",
                    help="whether to use the model name or the ")

args = parser.parse_args()

run(args.anomaly_detection, args.compare_to,
    args.model_type, args.model_names, args.model_mode, args.batch_size, args.id_datasets, args.datasets, args.metric, args.model_name_column)

#
# for bs in 5 1
# do
#   for method in gradients_L2_norms_gaussian typicality
#   do
#     python -m plots.anomaly_table \
#       glow cifar_long celeba svhn_working imagenet32 gtsrb_glow_continued \
#       --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
#       --datasets cifar10 celeba svhn imagenet32 gtsrb \
#       --anomaly_detection $method --batch_size $bs
#   done
# done