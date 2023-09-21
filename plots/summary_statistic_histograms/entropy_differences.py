
from os import path

import matplotlib.pyplot as plt

from plots.summary_statistic_histograms import (
    plot_summary_histograms, plot_summary_scatter, fetch_preliminaries, \
    get_save_dir_path, select_summary_stat_names, plot_fitted_distribution, label_getters,
    plot_fitted_distribution_scatter, get_input_var_xlabel
)


import argparse
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser


def run(model_type, model_names, model_mode, batch_size,
        id_datasets, ood_dataset_names, x_lim):


    fontsize = "xx-large"
    fig, axs = plt.subplots(nrows=len(model_names), figsize=(8, 12))

    bottom_ax = axs[-1]

    for model_name, id_dataset, ax in zip(model_names, id_datasets, axs):
        anomaly_detector, id_dataset_summary, ood_dataset_summaries = \
            fetch_preliminaries(model_type, model_name, model_mode, "likelihoods", batch_size,
                                id_dataset, ood_dataset_names, False)

        plot_summary_histograms(ax, id_dataset_summary, id_dataset,
                                ood_dataset_summaries, ood_dataset_names,
                                "-log p", x_lim=x_lim, take_log=False)
        ax.set_yticks([])
        ax.set_ylabel(id_dataset, fontsize=fontsize)

        if ax is not bottom_ax:
            ax.sharex(bottom_ax)
            # ax.set_xticklabels([])

    if model_type == "diffusion":
        xlabel = "$\\frac{p_{\\theta}(\\mathbf{x}_{0:1})}{q(\\mathbf{x}_{1} \\vert \\mathbf{x}_{0})}$"
    else:
        xlabel = "$\\frac{\\log_2 p(x)}{3 \\times 32 \\times 32}$"

    bottom_ax.set_xlabel(xlabel, fontsize=fontsize)

    handles, _ = ax.get_legend_handles_labels()

    fig.tight_layout()

    fig.legend(handles, ood_dataset_names, fontsize=fontsize)

    file_title = f"{model_type}_entropy_differences"

    filepath = path.join("entropy_difference_plots", file_title + ".png")

    plt.savefig(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])

    parser.add_argument("--x_lim", nargs=2, type=float, default=None,
                        help="the limits of the x-axis plot (defaults to min/max of the id dataset)")

    args = parser.parse_args()

    run(args.model_type, args.model_names, args.model_mode, args.batch_size,
        args.id_datasets, args.datasets, (-7, -1))
