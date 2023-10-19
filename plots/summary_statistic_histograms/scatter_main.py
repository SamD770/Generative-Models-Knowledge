import argparse

import matplotlib.pyplot as plt
import numpy as np

from plots.summary_statistic_histograms import plot_summary_histograms, plot_summary_scatter, fetch_preliminaries, get_input_var_xlabel
from plots.utils import save_plot

from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

# some random data
x = np.random.randn(1000)
y = np.random.randn(1000)


def run(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names,
        specified_layers=(('flow.layers.65.block.0.actnorm.bias', 512), ('flow.layers.99.invconv.upper', 2304))):

    _, id_dataset_summary, ood_dataset_summaries = \
        fetch_preliminaries(model_type, model_name, model_mode, anomaly_detection_name, batch_size,
                            id_dataset, ood_dataset_names, False)

    x_layer, y_layer = specified_layers

    boi = id_dataset_summary[x_layer]

    # code extended from https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html

    # Start with a square Figure.
    fig = plt.figure(figsize=(5, 5))
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no ticks
    ax_histx.set_xticks([])
    ax_histx.set_yticks([])
    ax_histy.set_xticks([])
    ax_histy.set_yticks([])

    plot_summary_histograms(ax_histx, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names, x_layer)
    plot_summary_histograms(ax_histy, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names, y_layer,
                            orientation='horizontal')

    input_var_xlabel = get_input_var_xlabel(batch_size)

    plot_summary_scatter(ax, id_dataset_summary, id_dataset,
                             ood_dataset_summaries, ood_dataset_names, x_layer, y_layer, n_scatter=1000, alpha=0.5)


    h, l = ax.get_legend_handles_labels()
    ax.legend(h, ["in-distribution", "out-of-distribution"])

    xlabel, ylabel = ["$\\log f_{\\mathbf{\\theta}_{" + layer_index + "}}(" + input_var_xlabel + ")  $"
                      for layer_index in "ij"]

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    save_plot("teaser_figure")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])

    args = parser.parse_args()

    for arg_model_name, arg_id_dataset in zip(args.model_names, args.id_datasets):
        run(args.model_type, arg_model_name, args.model_mode, args.anomaly_detection, args.batch_size,
                          arg_id_dataset, args.datasets)