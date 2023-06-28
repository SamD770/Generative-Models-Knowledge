import argparse
from command_line_utils import anomaly_method_parser, dataset_parser, model_parser

from torch import save

from models.utils import load_generative_model
from anomaly_methods.utils import anomaly_detection_methods_dict
from data.utils import get_dataset


def run(anomaly_method_name, batch_size, model_name, model_type, dataset_names, dataset_split, verbose=True):
    """Caches the summary statistics from the given anomaly method and model applied to the given datasets."""

    model = load_generative_model(model_type, model_name)

    anomaly_method_name = anomaly_detection_methods_dict[anomaly_method_name]

    anomaly_detector = anomaly_method_name.from_model(model)

    for dataset_name in dataset_names:

        dataset = get_dataset(dataset_name, dataset_split)
        dataset_summary = anomaly_detector.compute_summary_statistics(dataset, batch_size)

        filepath = anomaly_detector.summary_statistic_filepath(model_name, dataset_name, batch_size)

        if verbose:
            print("Saving summary statistics to:   ", filepath)
        save(dataset_summary, filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[anomaly_method_parser, dataset_parser, model_parser])
    args = parser.parse_args()

    run(
        args.anomaly_detection, args.batch_size, args.model_name, args.model_type, args.datasets, args.split
    )
