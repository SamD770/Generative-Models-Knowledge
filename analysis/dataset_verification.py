import torch

from data.datasets import get_CIFAR10, get_SVHN, get_celeba, get_imagenet32

print("getting cifar")
_, _, _, test_cifar = get_CIFAR10(False, "../", True)

print("getting svhn")
_, _, _, test_svhn = get_SVHN(False, "../", True)

print("getting celeba")
_, _, _, test_celeba = get_celeba("../")

print("getting imagenet")
_, _, _, test_imagenet = get_imagenet32("../")


for dataset, name in zip(
        [test_cifar, test_svhn, test_celeba, test_imagenet],
        ["cifar", "svhn", "celeba", "imagenet"]):
    print()
    print(f"statistics for {name}:")
    sample, label = dataset[69]
    print(f"type: {type(sample)}")
    print(f"shape: {sample.shape}")
    print(f"mean: {torch.mean(sample)}")
    print(f"range {(torch.min(sample), torch.max(sample))}")