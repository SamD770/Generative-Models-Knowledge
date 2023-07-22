import pandas as pd
import numpy as np

import argparse

from plots.utils import get_anomaly_scores
from command_line_utils import model_name_parser, anomaly_method_parser, dataset_parser


def run(anomaly_detection_name, model_names, id_datasets, dataset_names, batch_size):
    model_names = pd.Index(model_names)
    dataset_names = pd.Index(dataset_names)

    for model_name, id_dataset in zip(model_names, id_datasets):
        id_test_anomaly_scores, ood_anomaly_scores_list = get_anomaly_scores(anomaly_detection_name, batch_size,
                                                                             id_dataset, model_name,
                                                                             dataset_names)

    df = pd.DataFrame(columns=model_names, index=dataset_names)
    print(
        df.to_latex()
    )


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_name_parser, dataset_parser])

args = parser.parse_args()
run(args.anomaly_detection, args.model_names, args.id_datasets, args.datasets, args.batch_size)
