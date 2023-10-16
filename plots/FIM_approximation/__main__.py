import random
import argparse

import matplotlib.pyplot as plt
from matplotlib import gridspec

from command_line_utils import model_parser

from plots.utils import save_plot


from plots.FIM_approximation import compute_FIM, load_prerequisites, reparam_FIM

random.seed(1)  # Fixes the seed so randomly selected layers are verifiable


def run(model_type, model_name, sampling_model_name=None, n_layers=2, n_weights_per_layer=50):

    fim_store, model, n_selected_weights_list, sample_dataset = load_prerequisites(
        model_name, model_type, n_layers, n_weights_per_layer, sampling_model_name
    )

    FIM_windows = compute_FIM(fim_store, model, sample_dataset)

    # plot and beautify

    fig = plt.figure()

    width_ratios = [*n_selected_weights_list, 1] # add an extra row for the colorbar

    gs = gridspec.GridSpec(n_layers, n_layers+1,
                           height_ratios=n_selected_weights_list, width_ratios=width_ratios) # , wspace=0.0)

    # When plotting off-diagonals of the FIM, we compute F_ab / sqrt(F_aa Fb_b)

    diagonals = [
        FIM_windows[i][i].diagonal() for i in range(n_layers)
    ]

    for r, FIM_window_row in enumerate(FIM_windows):
        for c, matrix in enumerate(FIM_window_row):

            diagonal_rows = diagonals[r]
            diagonal_cols = diagonals[c]

            matrix = reparam_FIM(matrix, diagonal_cols, diagonal_rows)

            ax = plt.subplot(gs[r, c])

            cmap = ax.imshow(matrix.cpu(), cmap='RdBu',
                             vmin=-1, vmax=1)

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0:
                ax.set_title("layer " + "ijk"[c])  # need to add more index labels if want more plots.

            if c == 0:
                ax.set_ylabel("layer " + "ijk"[r])

    colorbar_ax = plt.subplot(gs[:, n_layers])

    cbar = fig.colorbar(cmap, cax=colorbar_ax)
    cbar.set_label("$\\frac{F_{\\alpha \\beta}}"
                   "{\\sqrt{F_{\\alpha\\alpha} F_{\\beta\\beta}}}$", fontsize="xx-large")

    plt.tight_layout()

    save_plot(f"{model_type}_{model_name}_FIM")


def concurrent_sort(key_list, value_list):
    """Sorts both lists according to key_list"""
    return zip(*sorted(zip(key_list, value_list), key=lambda x: x[0]))


parser = argparse.ArgumentParser(parents=[model_parser])

parser.add_argument("--sampling_model_name",
                    help="optional specification of a different model to draw samples from "
                         "(defaults to the evaluation model).")

parser.add_argument("--n_layers", type=int, default=2,
                    help="optional specification of a different model to draw samples from "
                         "(defaults to the evaluation model).")

args = parser.parse_args()


for arg_model_name in args.model_names:
    run(args.model_type, arg_model_name, args.sampling_model_name, n_layers=args.n_layers)


