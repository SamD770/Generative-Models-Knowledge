
from os import path

import matplotlib.pyplot as plt

from plots.summary_statistic_histograms import (
    plot_summary_histograms, fetch_preliminaries
)

from models.diffusion_model.model import extract_timesteps # only nice way to grep the timestep count out
from plots.utils import to_styled_dataset_name, save_plot

import argparse
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser, plotting_parser


def run(model_type, model_names, model_mode, batch_size,
        id_datasets, ood_dataset_names, x_lim, with_legend, with_train_dataset_labels):

    fontsize = "x-large"
    fig, axs = plt.subplots(nrows=len(model_names), figsize=(8, 10))

    title = f"{model_type} model".title()
    fig.suptitle(title, fontsize=fontsize)

    bottom_ax = axs[-1]

    for model_name, id_dataset, ax in zip(model_names, id_datasets, axs):
        anomaly_detector, id_dataset_summary, ood_dataset_summaries = \
            fetch_preliminaries(model_type, model_name, model_mode, "likelihoods", batch_size,
                                id_dataset, ood_dataset_names, False)

        plot_summary_histograms(ax, id_dataset_summary, id_dataset,
                                ood_dataset_summaries, ood_dataset_names,
                                "-log p", x_lim=x_lim, take_log=False)
        ax.set_yticks([])

        if with_train_dataset_labels:
            ax.set_ylabel(to_styled_dataset_name[id_dataset], fontfamily="monospace", fontsize=fontsize)

        if ax is not bottom_ax:
            ax.sharex(bottom_ax)
            # ax.set_xticklabels([])

    if model_type == "diffusion":
        _, timesteps = extract_timesteps(model_name)

        xlabel = "One-Step Negative Log Likelihood"
        # xlabel = "$L_{" + str(timesteps - 1) + "}(\\mathbf{x}_{1:5})$"
        file_title = f"{model_type}_{timesteps}_timesteps_entropy_differences_{batch_size}_batches"

    elif model_type == "glow":
        xlabel = "Negative Bits Per Dimension"
        # xlabel = "$\\frac{\\log_2 p(\\mathbf{x})}{3 \\times 32 \\times 32}$"
        file_title = f"{model_type}_entropy_differences_{batch_size}_batches"

    else:
        xlabel = f"xlabel not implemented for model type {model_type}"
        file_title = f"{model_type}_entropy_differences_{batch_size}_batches"

    bottom_ax.set_xlabel(xlabel, fontsize=fontsize)

    fig.tight_layout()

    if with_legend:
        handles, _ = ax.get_legend_handles_labels()

        styled_ood_dataset_names = [
            to_styled_dataset_name[dsn] for dsn in ood_dataset_names
        ]

        prop = {"family": "monospace"}

        fig.legend(handles, styled_ood_dataset_names, prop=prop, fontsize=fontsize)


    save_plot(file_title)

    # filepath = path.join("entropy_difference_plot", file_title + ".png")
    #
    # plt.savefig(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser, plotting_parser])

    parser.add_argument("--x_lim", nargs=2, type=float, default=None,
                        help="the limits of the x-axis plot (defaults to min/max of the id dataset)")

    parser.add_argument("--with_train_dataset_labels", action="store_true",
                        help="whether to add the train dataset labels to the y axes (default false)")

    args = parser.parse_args()

    run(args.model_type, args.model_names, args.model_mode, args.batch_size,
        args.id_datasets, args.datasets, args.x_lim, args.with_legend, args.with_train_dataset_labels)
