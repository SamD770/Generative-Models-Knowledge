import argparse

from models.utils import load_generative_model
from anomaly_methods.utils import anomaly_detection_methods_dict
from data.utils import to_dataset_wrapper

#  TODO: command line script for serialising summary statistics
