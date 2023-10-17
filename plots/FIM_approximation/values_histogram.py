import random
import argparse

import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np

from command_line_utils import model_parser

from plots.utils import save_plot


from plots.FIM_approximation import compute_FIM, load_prerequisites, device, reparam_FIM


random.seed(1)  # Fixes the seed so randomly selected layers are verifiable


def run(model_type, model_name, sampling_model_name=None,
        n_layers=100, n_weights_per_layer=1000, n_layers_plot=5):

    fim_store, model, n_selected_weights_list, sample_dataset = load_prerequisites(
        model_name, model_type, n_layers, n_weights_per_layer, sampling_model_name,
    )

    FIM_windows = compute_FIM(fim_store, model, sample_dataset)

    # plot and beautify

    fig = plt.figure()

    # When plotting off-diagonals of the FIM, we compute F_ab / sqrt(F_aa Fb_b)

    off_diagonal_averages = []

    for i in range(n_layers):
        matrix = FIM_windows[i][i]
        diagonal = matrix.diagonal()

        diagonal += 1e-14 # way to deal with "dead" weights giving NaN values

        matrix = reparam_FIM(matrix, diagonal, diagonal)

        n_weights = len(diagonal)

        # We subtract the identity matrix to remove the influence of the diagonal
        off_diagonals = matrix - torch.eye(n_weights, device=device)

        average_off_diagonal = off_diagonals.abs().nansum() / (n_weights * (n_weights - 1))

        if average_off_diagonal.isnan().any():
            print(f"{off_diagonals[:10, :10]=}")

        off_diagonal_averages.append(average_off_diagonal.item())

    # print(off_diagonal_averages)
    print(f"off diagonal average for {model_type} {model_name}:", sum(off_diagonal_averages) / n_layers)

    fig, ax = plt.subplots()

    # logbins = np.geomspace(1e-14, 1, 40)

    for i in range(n_layers_plot):
        matrix = FIM_windows[i][i]
        diagonal = matrix.diagonal()

        ax.hist(diagonal.log().cpu().numpy(), n_bins=10) # , bins=logbins)

        # ax.set_xscale("log")

    save_plot("diagonal size comparison")


parser = argparse.ArgumentParser(parents=[model_parser])

parser.add_argument("--sampling_model_name",
                    help="optional specification of a different model to draw samples from "
                         "(defaults to the evaluation model).")

args = parser.parse_args()


for arg_model_name in args.model_names:
    run(args.model_type, arg_model_name, args.sampling_model_name)


