import argparse
from command_line_utils import anomaly_method_parser, dataset_parser, model_parser

from torch import save

from models.utils import load_generative_model
from anomaly_methods.utils import anomaly_detection_methods_dict
from data.utils import get_dataset


def run(anomaly_method_name, batch_size, model_name, model_type, dataset_names, dataset_split, verbose=True):
    """Caches the summary statistics from the given anomaly method and model applied to the given datasets."""

    model = load_generative_model(model_type, model_name)

    anomaly_method = anomaly_detection_methods_dict[anomaly_method_name]

    anomaly_detector = anomaly_method.from_model(model)

    for dataset_name in dataset_names:

        cache_statistics(anomaly_detector, batch_size, dataset_name, dataset_split, model_name, verbose)

    if verbose:
        print("done")


def cache_statistics(anomaly_detector, batch_size, dataset_name, dataset_split, model_name, model_type,
                     filepath=None, verbose=True):

    if filepath is None:
        filepath = anomaly_detector.summary_statistic_filepath(model_type, model_name, dataset_name, batch_size)

    if verbose:
        print("Creating summary statistics file:   ", filepath)

    if anomaly_detector.model is None:
        anomaly_detector.model = load_generative_model(model_type, model_name)

    dataset = get_dataset(dataset_name, dataset_split)
    dataset_summary = anomaly_detector.compute_summary_statistics(dataset, batch_size)
    save(dataset_summary, filepath)


def load_statistics():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[anomaly_method_parser, dataset_parser, model_parser])
    args = parser.parse_args()

    for model_name_arg in args.model_names:
        run(
            args.anomaly_detection, args.batch_size, model_name_arg, args.model_type, args.datasets, args.split
        )
