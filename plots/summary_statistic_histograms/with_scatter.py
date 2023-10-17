
from os import path

import matplotlib.pyplot as plt

from plots.summary_statistic_histograms import (
    plot_summary_histograms, plot_summary_scatter, fetch_preliminaries,
    get_save_dir_path, select_summary_stat_names, plot_fitted_distribution,
    plot_fitted_distribution_scatter, get_input_var_xlabel
)

import argparse
from command_line_utils import model_parser, anomaly_method_parser, dataset_parser, plotting_parser

# plots three axes sequentially with two histogram plots and one scatter plot


def run(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names,
        fitted_distribution, with_legend=True):
    """Plots axes (one for each summary statistic) on one figure."""

    # Fetch cached statistics from the disk

    anomaly_detector, id_dataset_summary, ood_dataset_summaries = \
        fetch_preliminaries(model_type, model_name, model_mode, anomaly_detection_name, batch_size,
                            id_dataset, ood_dataset_names, fitted_distribution)

    save_dir_path = get_save_dir_path(model_name)

    # plot histograms of the data

    selected_stat_names = select_summary_stat_names(anomaly_detector.summary_statistic_names, 2)
    fig, axs = plt.subplots(ncols=3, figsize=(12, 4))

    file_title = f"gradients from two randomly selected layers of a {model_type} model trained on {id_dataset}"

    filepath = path.join(save_dir_path, file_title + ".png")

    print(f"creating: {filepath}")

    with open(filepath[:-4] + ".txt", "wt") as f:  # quick and dirty way to record the names used
        f.write(str(selected_stat_names))

    # fig.suptitle(figure_title)
    input_var_xlabel = get_input_var_xlabel(batch_size)

    # axes_labels = ["layer " + layer_index for layer_index in "ij"]

    x_labels = [
        "$\\log f_{\\mathbf{\\theta}_{" + layer_index +"}}(" + input_var_xlabel +")  $" for layer_index in "ij"
    ]

    fontsize = "medium"

    histogram_axs = axs[:-1]
    scatter_ax = axs[-1]

    for stat_name, ax, x_label in zip(selected_stat_names, histogram_axs, x_labels):

        plot_summary_histograms(ax, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names, stat_name)

        if fitted_distribution:
            plot_fitted_distribution(ax, anomaly_detector, stat_name)
        ax.set_yticks([])
        ax.set_xlabel(x_label, fontsize=fontsize)

    plot_summary_scatter(scatter_ax, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names,
                         selected_stat_names[0], selected_stat_names[1])

    # Grab the labels from the last axes to prevent label duplication

    left_histogram_ax = histogram_axs[0]
    y_label = f"{model_type} model".title()
    left_histogram_ax.set_ylabel(y_label, fontsize=fontsize)

    if with_legend: # only add the legend to the first plot created
        fig.legend(*left_histogram_ax.get_legend_handles_labels(), fontsize=fontsize)

    scatter_ax.set_xlabel(x_labels[0], fontsize=fontsize)
    scatter_ax.set_ylabel(x_labels[1], fontsize=fontsize)

    if fitted_distribution:
        plot_fitted_distribution_scatter(scatter_ax, anomaly_detector, selected_stat_names[0], selected_stat_names[1])

    fig.tight_layout()

    plt.savefig(filepath)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser, plotting_parser])

    parser.add_argument("--fitted_distribution", action="store_true",
                        help="whether plot the fitted distribution, if it exists, "
                             "with the summary statistics (default False)")

    args = parser.parse_args()

    for arg_model_name, arg_id_dataset in zip(args.model_names, args.id_datasets):

        run(args.model_type, arg_model_name, args.model_mode, args.anomaly_detection, args.batch_size,
            arg_id_dataset, args.datasets, args.fitted_distribution, args.with_legend)