import sys
from os import path
from data.utils import dataset_names
from models.utils import model_classes

import argparse

RUNNING_MODULE_DIR, _ = path.split(sys.argv[0])


# Define parent parsers

dataset_parser = argparse.ArgumentParser(add_help=False)
dataset_parser.add_argument("-ds", "--datasets", nargs="+", choices=dataset_names,
                            help="The dataset(s) to run the plot on.")

model_parser = argparse.ArgumentParser(add_help=False)
model_parser.add_argument("--model-type", choices=model_classes,
                          help="The type of model to run the plot on.")

model_parser.add_argument("--model-name",
                          help="The name of the model to run the plot on.")