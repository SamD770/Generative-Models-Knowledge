import argparse

from plots.anomaly_table import get_dataframe, model_name_formatter, dataset_name_formatter, metric_dict, \
    get_performance_stats
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser


def run(model_type, model_names, model_mode, anomaly_detection_name, batch_size, id_datasets, dataset_names,
        metric_name, model_name_column):

    df = get_dataframe(anomaly_detection_name, batch_size, dataset_names, id_datasets, metric_name, model_mode,
                       model_names, model_name_column, model_type)

    avg_performance, quantiles, stdev_performance = get_performance_stats(df)

    title = f"{metric_name} values for {anomaly_detection_name}, batch size {batch_size} applied to {model_type} " \
            f"in {model_mode} mode, " \
            f"\\newline average performance: {avg_performance:.4f} (stdev: {stdev_performance:.4f})" \
            f"\\newline 25/50/75 quantiles: {quantiles[0]:.4f} / {quantiles[1]:.4f} / {quantiles[2]:.4f}"

    caption = title.replace("_", "\\_")

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
        formatter=dataset_name_formatter,
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
parser.add_argument("--model_name_column", action="store_true",
                    help="whether to use the model name or the ")

args = parser.parse_args()
run(args.model_type, args.model_names, args.model_mode,
    args.anomaly_detection, args.batch_size, args.id_datasets, args.datasets, args.metric, args.model_name_column)
