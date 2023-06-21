import sys
from os import path
from data.utils import dataset_names
from models.utils import model_classes

from torchvision.utils import make_grid


import argparse

RUNNING_MODULE_DIR, _ = path.split(sys.argv[0])


# Define parent parsers

dataset_parser = argparse.ArgumentParser(add_help=False)
dataset_parser.add_argument("-ds", "--datasets", nargs="+", choices=dataset_names,
                            help="The dataset(s) to run the plot on.")

model_parser = argparse.ArgumentParser(add_help=False)
model_parser.add_argument("model_type", choices=model_classes,
                          help="The type of model to run the plot on.")

model_parser.add_argument("model_names", nargs="+",
                          help="The name of the model to run the plot on.")


def grid_from_imgs(img_seq):
    """Takes a sequence of images and returns a grid with  that can be plotted using plt.imshow"""
    grid = make_grid(img_seq, nrow=8)

    grid = grid.permute(1, 2, 0)

    if grid.min() < 0:  # To account for the fact that the colour datasets are scaled (-0.5, 0.5)
        grid += 0.5

    return grid