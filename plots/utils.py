import sys
from os import path

from matplotlib import pyplot as plt


from torchvision.utils import make_grid

RUNNING_MODULE_DIR, _ = path.split(sys.argv[0])


# Define parent parsers

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
