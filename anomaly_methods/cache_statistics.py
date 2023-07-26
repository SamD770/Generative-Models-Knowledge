import argparse
from command_line_utils import anomaly_method_parser, dataset_parser, model_parser

from torch import save

from models.utils import load_generative_model
from anomaly_methods.utils import anomaly_detection_methods_dict
from data.utils import get_dataset


def run(model_type, model_name, anomaly_method_name, batch_size, dataset_names, dataset_split, verbose=True):
    """Caches the summary statistics from the given anomaly method and model applied to the given datasets."""

    model = load_generative_model(model_type, model_name)

    anomaly_method = anomaly_detection_methods_dict[anomaly_method_name]

    anomaly_detector = anomaly_method.from_model(model)

    for dataset_name in dataset_names:

        filepath = anomaly_detector.summary_statistic_filepath(model_type, model_name, dataset_name, batch_size)

        cache_statistics(filepath, anomaly_detector, batch_size, dataset_name, dataset_split)

    if verbose:
        print("done")


def cache_statistics(filepath, anomaly_detector, batch_size, dataset_name, dataset_split="test", verbose=True):

    if verbose:
        print("Creating summary statistics file:   ", filepath)

    dataset = get_dataset(dataset_name, dataset_split)
    dataset_summary = anomaly_detector.compute_summary_statistics(dataset, batch_size)
    save(dataset_summary, filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[anomaly_method_parser, dataset_parser, model_parser])
    args = parser.parse_args()

    for model_name_arg in args.model_names:
        run(args.model_type, model_name_arg, args.anomaly_detection, args.batch_size, args.datasets, args.split)
