from plots.anomaly_table import get_dataframe, metric_dict
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

import argparse
import pandas as pd

def run(model_type, model_names, model_mode, batch_size, id_datasets, dataset_names,
        metric_name, model_name_column):

    method_1 = "typicality"
    method_2 = "gradients_L2_norms_gaussian"

    df_typicality = get_dataframe(method_1, batch_size, dataset_names,
                                  id_datasets, metric_name, model_mode, model_names, model_name_column, model_type)

    df_gradients = get_dataframe(method_2, batch_size, dataset_names,
                                 id_datasets, metric_name, model_mode, model_names, model_name_column, model_type)

    df_comparison = df_gradients > df_typicality

    # A silly little hack to store the float values which we cant to be bold

    bold_flag_dict = {
        performance: greater_than for performance, greater_than in zip(df_gradients.stack(), df_comparison.stack())
    } | {
        performance: not greater_than for performance, greater_than in zip(df_typicality.stack(), df_comparison.stack())
    }

    def bold_formatter(performance):

        if pd.isna(performance):
            return "-"

        bold_flag = bold_flag_dict[performance]

        performance = format(performance, ".4f")

        if bold_flag:
            performance = "\\B{" + performance + "}"

        return performance

    styler = df_gradients.style

    styler = styler.format(
        formatter=bold_formatter
    )

    table_latex = styler.to_latex(
      hrules=True,
      position="H",
      column_format="l | r r r r r"
    )


    print(table_latex)

    df_comparison = (df_gradients > df_typicality)



parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])
parser.add_argument("--metric", choices=metric_dict.keys(),
                    help="The metric by which to measure the success of the anomaly detection method", default="auc")
parser.add_argument("--model_name_column", action="store_true",
                    help="whether to use the model name or the ")

args = parser.parse_args()

run(args.model_type, args.model_names, args.model_mode, args.batch_size, args.id_datasets, args.datasets, args.metric, args.model_name_column)

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