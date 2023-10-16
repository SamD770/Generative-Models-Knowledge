from data.utils import get_dataset
from torch.nn.functional import pad
import numpy as np
from scipy.special import gammaln

from math import exp, log


def total_variation(img):
    vertical_diffs = img[1:] - img[:-1]
    horizontal_diffs = img[:, 1:] - img[:, :-1]

    vertical_variation = vertical_diffs.abs().sum()
    horizontal_variation = horizontal_diffs.abs().sum()

    tv = vertical_variation + horizontal_variation
    return tv.item()


def snake_variation(img):
    """Computes the total variation given by taking a snake through the image pixels"""
    img = img[0]
    horizontal_diffs = img[:, 1:] - img[:, :-1]

    horizontal_variation = horizontal_diffs.__abs__().sum()

    n_1, n_2 = img.shape

    if n_2 % 2 == 0:
        vertical_left_edge_diffs = img[1:(n_2-1):2, 0] - img[2:(n_2-1):2, 0]
        vertical_right_edge_diffs = img[0:n_2:2, -1] - img[1:n_2:2, -1]

        vertical_edge_variation = vertical_left_edge_diffs.__abs__().sum() + vertical_right_edge_diffs.__abs__().sum()
    else:
        raise ValueError("snake_variation not implemented for images of odd dimensionality")

    return horizontal_variation + vertical_edge_variation


if __name__ == "__main__":
    my_dataset = get_dataset("MNIST")

    snake_vars = []
    measure_bounds = []

    for i in range(len(my_dataset)):
        x, y = my_dataset[i]

        # print("total var:", total_var)

        snake_var = snake_variation(x)
        # print("snake var:", snake_var)

        snake_vars.append(snake_var)

        d = 28*28
        log_lebesgue_measure_bound = d * np.log(2 * snake_var) - gammaln(d+1)
        measure_bounds.append(log_lebesgue_measure_bound)

        # print("measure bound:", log_lebesgue_measure_bound)

    max_snake_var = max(*snake_vars)
    print(f"{max_snake_var=}")

    max_measure_bound = max(*measure_bounds)

    print(measure_bounds[:30])

    print("max_measure_bound: ", max_measure_bound)
    print("measure_boud/dim", max_measure_bound/(28*28))

    max_measure_bound_ten_power = max_measure_bound/log(10.)

    print("log_10 E(alpha)", max_measure_bound_ten_power)
