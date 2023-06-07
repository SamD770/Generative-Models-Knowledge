from data.utils import (
    SampleDataset,
    get_vanilla_dataset,
)

from models.utils import load_generative_model, compute_nll

import matplotlib.pyplot as plt

from torch.utils.data import Subset
import argparse

parser = argparse.ArgumentParser(
    description="Compares the likelihood histograms for the given models and training datasets")

model_class = "vae"

model_names = ["cifar_long", "svhn_working", "imagenet32", "celeba"]

dataset_names = ["cifar", "svhn", "imagenet32", "celeba"]


sample_dataset = False

draw_cross_entropy = False


if sample_dataset:
    datasets = [
        SampleDataset(model) for model in models
    ]
else:
    datasets = [
        get_vanilla_dataset(dataset_name) for dataset_name in dataset_names
    ]


fig, axs = plt.subplots(nrows=4, ncols=1)

top_likelihood_ax = axs[0]  # , top_sample_ax = axs[0]
last_likelihood_ax = axs[-1]

for model, model_name, ax in zip(models, dataset_names, axs):

    likelihood_ax = ax  # , sample_ax = ax

    likelihood_ax.sharex(top_likelihood_ax)

    likelihood_ax.set_xticklabels([])
    likelihood_ax.set_yticks([])

    likelihood_ax.set_ylabel(f"{model_name}")

    for dataset, dataset_name in zip(datasets, dataset_names):
        nlls = compute_nll(Subset(dataset, range(512 * 10)), model)
        nlls = nlls.clamp(max=11)

        print("-", end="")

        if likelihood_ax is top_likelihood_ax:
            label = dataset_name
        else:
            label = None

        likelihood_ax.hist(
            -nlls.numpy(), density=True, range=(-7, -1), bins=30, alpha=0.6, label=label
        )

        if draw_cross_entropy:
            cross_entropy = nlls.mean()
            likelihood_ax.vline(cross_entropy)  # should check this function.
        print("-")


last_likelihood_ax.set_xlabel("log likelihood ")

fig.legend(title="evaluation dataset")
# fig.tight_layout()

plt.savefig(
    "./anomaly_methods/plots/seminal_paper_recreations/likelihood_histogram_comparison_refined.png"
)

print("done")
