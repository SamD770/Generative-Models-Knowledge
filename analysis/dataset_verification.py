import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from random import randint

from data.datasets import get_CIFAR10, get_SVHN, get_celeba, get_imagenet32

print("getting cifar")
_, _, _, test_cifar = get_CIFAR10(False, "../", True)

print("getting svhn")
_, _, _, test_svhn = get_SVHN(False, "../", True)

print("getting celeba")
_, _, _, test_celeba = get_celeba("../")

print("getting imagenet")
_, _, _, test_imagenet = get_imagenet32("../")

SAMPLE_COUNT = 32


for dataset, name in zip(
        [test_cifar, test_svhn, test_celeba, test_imagenet],
        ["cifar", "svhn", "celeba", "imagenet"]):
    print()
    print(f"statistics for {name}:")
    sample, label = dataset[69]
    tot_samples = len(dataset)
    print(f"length: {tot_samples}")
    print(f"type: {type(sample)}")
    print(f"shape: {sample.shape}")
    print(f"mean: {torch.mean(sample)}")
    print(f"range {(torch.min(sample), torch.max(sample))}")
    sample = torch.permute(sample, (1, 2, 0)) + 0.5

    samples = []

    for _ in range(SAMPLE_COUNT):
        index = randint(0, tot_samples)
        sample, _ = dataset[index]
        samples.append(sample)

    grid = make_grid(samples, nrow=8)

    grid = grid.permute(1, 2, 0) + 0.5

    title = f"samples from {name} dataset"

    plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    plt.savefig("plots/sample_plots/" + title + ".png")

print("done")