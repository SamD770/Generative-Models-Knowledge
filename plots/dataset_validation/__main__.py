
import torch
import matplotlib.pyplot as plt
from random import randint

from data.utils import get_test_dataset
from plots.utils import dataset_parser, grid_from_imgs, save_plot

import argparse


parser = argparse.ArgumentParser(parents=[dataset_parser])


SAMPLE_COUNT = 32


def run(dataset_name):
    print(dataset_name)
    dataset = get_test_dataset(dataset_name)
    print(f"statistics for {dataset_name}:")
    sample, label = dataset[1729]

    tot_samples = len(dataset)
    print(f"length: {tot_samples}")
    print(f"type: {type(sample)}")

    print(f"shape: {sample.shape}")

    print(f"mean: {torch.mean(sample)}")
    print(f"range {(torch.min(sample), torch.max(sample))}")

    samples = []

    for _ in range(SAMPLE_COUNT):
        index = randint(0, tot_samples)
        sample, _ = dataset[index]
        samples.append(sample)

    grid = grid_from_imgs(samples)

    title = f"samples from {dataset_name} dataset"

    # plt.title(title)
    plt.imshow(grid)
    plt.axis("off")

    save_plot(title)


args = parser.parse_args()

for dataset_name in args.datasets:
    run(dataset_name)
