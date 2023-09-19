from data.utils import get_dataset
from torch.nn.functional import pad
import numpy as np
from scipy.stats import gammaln

def total_variation(img):
    vertical_diffs = img[1:] - img[:-1]
    horizontal_diffs = img[:, :1] - img[:, :-1]

    vertical_variation = vertical_diffs.abs().sum()
    horizontal_variation = horizontal_diffs.abs().sum()

    tv = vertical_variation + horizontal_variation
    return tv.item()



if __name__ == "__main__":
    my_dataset = get_dataset("MNIST")
    x, y = my_dataset[0]

    total_var = total_variation(x)
    print("total var:", total_var)

    d = 28*28
    lebesgue_measure_bound = d * np.log(2 * total_var) - gammaln(d+1)
    gammaln(x)
