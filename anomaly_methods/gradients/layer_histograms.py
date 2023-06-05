from .gradient_utils import *

import matplotlib.pyplot as plt

import random


def layer_histograms(batch_size, model_name, id_dataset, ood_datasets):
    id_norms, ood_norms_list = get_norms(
        batch_size, model_name, id_dataset, ood_datasets
    )

    all_layers = list(id_norms.keys())

    print("layer count:", len(all_layers))

    n_layers = 2

    rand_layers = [random.choice(all_layers) for _ in range(n_layers)]

    # rand_layers = ["flow.layers.40.actnorm.bias", "flow.layers.90.invconv.lower"]

    print("using layer names: ", *rand_layers)

    fig, axs = plt.subplots(1, n_layers, figsize=(9, 3))

    labeled = False

    save_title = (
        f"Gradient histogram ({model_name}, batch size {batch_size}, {rand_layers})"
    )

    scatter_vals = [[], []]
    labels = []

    n_scatter = 200

    axis_labels = [r"$\log f_{\theta_{" + s + r"}}(x_{1:B})$" for s in "ij"]
    axis_titles = [f"layer {s}" for s in "ij"]

    for value_list, layer_name, ax, axis_label, axis_title in \
            zip(scatter_vals, rand_layers, axs, axis_labels, axis_titles):

        ax.set_yticks([])

        log_id_gradients = torch.log(id_norms[layer_name])

        if labeled:
            label = None
        else:
            label = f"in distribution {id_dataset}"

        ax.set_xlabel(axis_label)
        ax.set_title(axis_title)
        print("setting title", axis_title)


        value_list.append(log_id_gradients[:n_scatter])
        labels.append(label)

        ax.hist(log_id_gradients.numpy(), label=label, density=True, alpha=0.6, bins=40)

        for ood_norms, ood_dataset_name in zip(ood_norms_list, ood_datasets):
            if labeled:
                label = None
            else:
                label = f"out-of-distribution {ood_dataset_name}"

            log_ood_gradients = torch.log(ood_norms[layer_name])

            labels.append(label)
            value_list.append(log_ood_gradients[:n_scatter])

            ax.hist(
                log_ood_gradients.numpy(), label=label, density=True, alpha=0.6, bins=40
            )

        labeled = True

    fig.legend(loc='upper center')

    plt.tight_layout()

    plt.savefig(f"./analysis/plots/gradient_plots/{save_title}.png")

    nombres = [id] + ood_datasets

    fig, ax = plt.subplots()

    print()
    print()

    for x, y in zip(*scatter_vals):
        ax.scatter(x, y, marker=".")

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    plt.savefig(f"./analysis/plots/gradient_plots/{save_title}_scatter.png")


if __name__ == "__main__":
    ood_datasets = ["Omniglot", "MNIST"]
    id_dataset = "FashionMNIST"
    model_name = "FashionMNIST_stable"

    layer_histograms(5, model_name, id_dataset, ood_datasets)

    # dataset_names = ["cifar", "svhn", "imagenet32", "celeba"]
    # for model_name, training_dataset in zip(["cifar_long", "svhn_working", "imagenet32", "celeba"],
    #                                    dataset_names):
    #     ood_datasets = copy(dataset_names)
    #     ood_datasets.remove(training_dataset)
    #     layer_histograms(5, model_name, training_dataset, ood_datasets)
