import argparse

from data.utils import dataset_names
from models.utils import model_classes
from anomaly_methods.utils import anomaly_detection_methods


dataset_parser = argparse.ArgumentParser(add_help=False)
dataset_parser.add_argument("-ds", "--datasets", nargs="+", choices=dataset_names.union({"cifar"}),
                            help="The dataset(s) to run the plot on.")

dataset_parser.add_argument("--split", choices=["train", "test"],
                            help="the dataset split to use", default="test")

model_name_parser = argparse.ArgumentParser(add_help=False)
model_name_parser.add_argument("model_names", nargs="+",
                               help="The name of the model to run the plot on.")

model_type_parser = argparse.ArgumentParser(add_help=False)
model_type_parser.add_argument("model_type", choices=model_classes,
                               help="The type of model to run the plot on.")

model_parser = argparse.ArgumentParser(add_help=False, parents=[model_type_parser, model_name_parser])

anomaly_method_parser = argparse.ArgumentParser(add_help=False)
anomaly_method_parser.add_argument("--anomaly_detection", choices=anomaly_detection_methods,
                                   help="the anomaly detection method to use")

anomaly_method_parser.add_argument("--id_datasets", nargs="+", choices=dataset_names.union({"cifar"}),
                                   help="the in-distribution dataset(s)")

anomaly_method_parser.add_argument("--batch_size", type=int,
                                   help="the batch size used with the anomaly detection method")

