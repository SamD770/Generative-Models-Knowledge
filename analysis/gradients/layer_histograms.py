
from gradient_utils import *

import matplotlib.pyplot as plt


def layer_histograms(batch_size, model_name, id_dataset, ood_datasets):
    id_norms, ood_norms_list = get_norms(batch_size, model_name, id_dataset, ood_datasets)
    for n in [1, 10, 40, 80, 100]:
        plt.figure(figsize=(20, 10))
        layer_name = f"flow.layers.{n}.actnorm.bias"

        title = f"Gradient histogram ({model_name}, batch size {batch_size}, {layer_name})"

        plt.title(title)
        plt.xlabel("$\log L^2$ norm")

        log_id_gradients = torch.log(id_norms[layer_name])

        plt.hist(log_id_gradients.numpy(),
                 label=f"in distribution {id_dataset}", density=True, alpha=0.6, bins=40)

        for ood_norms, ood_dataset_name in zip(ood_norms_list, ood_datasets):
            log_ood_gradients = torch.log(ood_norms[layer_name])
            plt.hist(log_ood_gradients.numpy(),
                     label=f"out-of-distribution {ood_dataset_name}", density=True, alpha=0.6, bins=40)

        plt.legend()

        plt.savefig(f"plots/{title}.png")




if __name__ == "__main__":
    pass