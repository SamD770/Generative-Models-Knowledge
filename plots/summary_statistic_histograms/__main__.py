from plots.utils import RUNNING_MODULE_DIR

from plots.summary_statistic_histograms import plot_summary_histograms, plot_summary_scatter, fetch_preliminaries, \
    label_getters, get_save_dir_path, select_summary_stat_names, plot_fitted_distribution

import argparse

from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

import matplotlib.pyplot as plt

from os import path


# TODO: move these functions to separate files


def run_multi_figures(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names,
                      fitted_distribution, n_statistics_plot, x_lim):
    """Plots on one figure and axes for each summary statistic."""

    # Fetch cached statistics from the disk

    anomaly_detector, id_dataset_summary, ood_dataset_summaries = \
        fetch_preliminaries(model_type, model_name, model_mode, anomaly_detection_name, batch_size,
                            id_dataset, ood_dataset_names, fitted_distribution)

    save_dir_path = get_save_dir_path(model_name)

    # plot histograms of the data

    selected_stat_names = select_summary_stat_names(anomaly_detector.summary_statistic_names, n_statistics_plot)

    for stat_name in selected_stat_names:
        label_getter = label_getters[anomaly_detection_name]

        file_title, figure_title, xlabel = label_getter(
            model_type, model_name, batch_size, id_dataset, anomaly_detector.summary_statistic_names, n_statistics_plot,
            single_figure=False, stat_name=stat_name
        )

        # file_title = f"{stat_name} gradient histogram"

        filepath = path.join(save_dir_path, file_title + ".png")

        print(f"creating: {filepath}")

        # figure_title = f"gradients from 1 layer out of {len(anomaly_detector.summary_statistic_names)} in a {model_type} model"
        fig, ax = plt.subplots()
        # fig.suptitle(figure_title)

        try:
            plot_summary_histograms(ax, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names, stat_name, x_lim)
        except ValueError:
            continue

        if args.fitted_distribution:
            plot_fitted_distribution(ax, anomaly_detector, stat_name)

        ax.set_yticks([])
        ax.set_xlabel(xlabel) # "$ \\log f_{\\mathbf{\\theta}_\\ell}(\\mathbf{x}_{1:B})  $")

        ax.legend()

        plt.savefig(filepath)


def run_single_figure(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names,
                      fitted_distribution, n_statistics_plot, x_lim):
    """Plots axes (one for each summary statistic) on one figure."""

    # Fetch cached statistics from the disk

    anomaly_detector, id_dataset_summary, ood_dataset_summaries = \
        fetch_preliminaries(model_type, model_name, model_mode, anomaly_detection_name, batch_size,
                            id_dataset, ood_dataset_names, fitted_distribution)

    print("\n",  model_type, "n_statistics: ", len(anomaly_detector.summary_statistic_names), "\n")

    save_dir_path = get_save_dir_path("Extra gradient histograms")

    # plot histograms of the data

    if n_statistics_plot is None:
        raise ValueError("Need to specify n_statistics if using one figure.")

    selected_stat_names = select_summary_stat_names(anomaly_detector.summary_statistic_names, n_statistics_plot)
    fig, axs = plt.subplots(ncols=n_statistics_plot, figsize=(16, 12 / n_statistics_plot))

    label_getter = label_getters[anomaly_detection_name]

    file_title, figure_title, xlabel = label_getter(
        model_type, model_name, batch_size, id_dataset, anomaly_detector.summary_statistic_names, n_statistics_plot,
        single_figure=True, stat_name=None
    )

    # file_title = f"{model_type} {model_name} gradient histogram"

    filepath = path.join(save_dir_path, file_title + ".png")

    with open(filepath[:-4] + ".txt", "wt") as f: # quick and dirty way to record the names used
        f.write(str(selected_stat_names))

    print(f"creating: {filepath}")

    # figure_title = f"gradients from {n_statistics_plot} randomly selected layers out of " \
    #                f"{len(anomaly_detector.summary_statistic_names)} in a {model_type} model trained on {id_dataset}"
    # fig.suptitle(figure_title)


    for stat_name, ax in zip(selected_stat_names, axs):

        plot_summary_histograms(ax, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names, stat_name,
                                ) # fit_id_x_lim=True, x_lim=x_lim)

        if args.fitted_distribution:
            plot_fitted_distribution(ax, anomaly_detector, stat_name)

        ax.set_yticks([])
        ax.set_xlabel(xlabel) # "$ \\log f_{\\mathbf{\\theta}_\\ell}(\\mathbf{x}_{1:B})  $")

    leftmost_ax = axs[0]
    leftmost_ax.set_ylabel(f"{model_type} model trained on {id_dataset}")

    # Grab the labels from the last axes to prevent label duplication
    fig.legend(*ax.get_legend_handles_labels())
    fig.tight_layout()

    plt.savefig(filepath)


parser = argparse.ArgumentParser(parents=[anomaly_method_parser, model_parser, dataset_parser])

parser.add_argument("--fitted_distribution", action="store_true",
                    help="whether plot the fitted distribution, if it exists, "
                         "with the summary statistics (default False)")

parser.add_argument("--n_statistics", type=int, default=None,
                    help="number of statistics to randomly select and plot (defaults to printing all)")

parser.add_argument("--same_figure", action="store_true",
                    help="Whether or not to plot the statistics for the same model on the same figure.")

parser.add_argument("--x_lim", nargs=2, type=float, default=None,
                    help="the limits of the x-axis plot (defaults to min/max of the id dataset)")

args = parser.parse_args()

for arg_model_name, arg_id_dataset in zip(args.model_names, args.id_datasets):
    if args.same_figure:
        run_single_figure(args.model_type, arg_model_name, args.model_mode, args.anomaly_detection, args.batch_size,
                          arg_id_dataset, args.datasets, args.fitted_distribution, args.n_statistics, args.x_lim)
    else:
        run_multi_figures(args.model_type, arg_model_name, args.model_mode, args.anomaly_detection, args.batch_size,
                          arg_id_dataset, args.datasets, args.fitted_distribution, args.n_statistics, args.x_lim)



