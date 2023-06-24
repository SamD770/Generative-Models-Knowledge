import sys
from os import path

from matplotlib import pyplot as plt

from data.utils import dataset_names
from models.utils import model_classes
from anomaly_methods.utils import anomaly_detection_methods

from torchvision.utils import make_grid


import argparse

RUNNING_MODULE_DIR, _ = path.split(sys.argv[0])


# Define parent parsers

dataset_parser = argparse.ArgumentParser(add_help=False)
dataset_parser.add_argument("-ds", "--datasets", nargs="+", choices=dataset_names,
                            help="The dataset(s) to run the plot on.")


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

anomaly_method_parser.add_argument("--id_dataset", choices=dataset_names,
                                   help="the in-distribution dataset")

anomaly_method_parser.add_argument("--batch_size", type=int,
                                   help="the batch size used with the anomaly detection method")


def grid_from_imgs(img_seq):
    """Takes a sequence of images and returns a grid with  that can be plotted using plt.imshow"""
    grid = make_grid(img_seq, nrow=8)

    grid = grid.permute(1, 2, 0)

    if grid.min() < 0:  # To account for the fact that the colour datasets are scaled (-0.5, 0.5)
        grid += 0.5

    return grid


def save_plot(title):
    save_filepath = path.join(RUNNING_MODULE_DIR, title + ".png")
    plt.savefig(save_filepath)
