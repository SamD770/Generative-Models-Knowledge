import pandas as pd
import numpy as np

import argparse

from plots.utils import save_plot
from command_line_utils import model_name_parser, anomaly_method_parser, dataset_parser
from anomaly_methods.utils import anomaly_detection_methods_dict


def run(anomaly_detection_name, model_names, id_datasets, ood_dataset_names, batch_size):
    model_names = pd.Index(model_names)
    dataset_names = pd.index(id_datasets)

    df = pd.DataFrame(columns=model_names, index=dataset_names)
    print(
        df.to_latex()
    )


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_name_parser, dataset_parser])

args = parser.parse_args()
run(args.anomaly_detection, args.model_names, args.id_datasets, args.datasets, args.batch_size)
