import random
import argparse

import matplotlib.pyplot as plt
from matplotlib import gridspec

from command_line_utils import model_parser

from plots.utils import save_plot


from plots.FIM_approximation import compute_FIM, load_prerequisites

random.seed(1)  # Fixes the seed so randomly selected layers are verifiable


def run(model_type, model_name, sampling_model_name=None, n_layers=4, n_weights_per_layer=50):

    fim_store, model, n_selected_weights_list, sample_dataset = load_prerequisites(
        model_name, model_type, n_layers, n_weights_per_layer, sampling_model_name,
    )

    FIM_windows = compute_FIM(fim_store, model, sample_dataset)

    # plot and beautify

    fig, axs = plt.subplots(ncols=n_layers, figsize=(16, 16/n_layers))

    for i, ax in enumerate(axs):
        matrix = FIM_windows[i][i]
        cmap = ax.imshow(matrix.cpu(), cmap='RdBu',
                         vmin=-matrix.max(), vmax=matrix.max())

        cbar = fig.colorbar(cmap, ax=ax)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    save_plot(f"{model_type}_{model_name}_FIMs_raw")


parser = argparse.ArgumentParser(parents=[model_parser])

parser.add_argument("--sampling_model_names", nargs="+",
                    help="optional specification of a different model to draw samples from "
                         "(defaults to the evaluation model).")

args = parser.parse_args()

if args.sampling_model_names:
    sampling_models = args.sampling_model_names
else:
    sampling_models = args.model_names

for arg_model_name, arg_sampling_model_name in zip(args.model_names, sampling_models):
    run(args.model_type, arg_model_name, arg_sampling_model_name)


