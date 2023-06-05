import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from random import randint

from data.datasets import (
    get_CIFAR10,
    get_SVHN,
    get_celeba,
    get_imagenet32,
    get_MNIST,
    get_FashionMNIST,
    get_Omniglot,
    get_flipped_Omniglot,
)

# print("getting cifar")
# _, _, _, test_cifar = get_CIFAR10(False, "../", True)
#
# print("getting svhn")
# _, _, _, test_svhn = get_SVHN(False, "../", True)
#
# print("getting celeba")
# _, _, _, test_celeba = get_celeba("../")

# print("getting imagenet")
# _, _, _, test_imagenet = get_imagenet32("../")
#
#
print("getting MNIST")
_, _, _, test_mnist = get_MNIST("./")

print("getting fashionMNIST")
_, _, _, test_fashion_mnist = get_FashionMNIST("./")

print("getting Omniglot")
_, _, _, test_Omniglot = get_Omniglot("./")

print("getting flipped Omniglot")
_, _, _, test_flipped_Omniglot = get_flipped_Omniglot("./")


SAMPLE_COUNT = 32


for dataset, name in zip(
    [test_mnist, test_fashion_mnist, test_Omniglot, test_flipped_Omniglot],
    ["MNIST", "FashionMNIST", "Omniglot", "flipped_Omniglot"],
):
    print()
    print(f"statistics for {name}:")
    sample, label = dataset[69]
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

    grid = make_grid(samples, nrow=8)

    grid = grid.permute(1, 2, 0)

    title = f"samples from {name} dataset"

    # plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    plt.savefig("anomaly_methods/plots/sample_plots/" + title + ".png")

print("done")
