import random

from plots.summary_statistic_histograms import plot_summary_histograms
from plots.utils import RUNNING_MODULE_DIR, get_dataset_summmaries

import torch

import argparse

from anomaly_methods.utils import anomaly_detection_methods_dict
from anomaly_methods.gradients.L2_norms import DistributionFittingL2Norm

from command_line_utils import model_parser, anomaly_method_parser, dataset_parser

import matplotlib.pyplot as plt

from os import path
from os import makedirs


def get_input_var_xlabel(batch_size):
    if batch_size == 1:
        input_var_xlabel = "\\mathbf{x}"
    else:
        input_var_xlabel = "\\mathbf{x}_{1:" + str(batch_size) + "}"
    return input_var_xlabel


def gradients_L2_norms_labels(model_type, model_name, batch_size, id_dataset, n_statistics_method, n_statistics_plot, single_figure, stat_name=None):

    if single_figure:
        file_title = f"{model_type} {model_name} gradient histogram"
        figure_title = f"gradients from {n_statistics_plot} randomly selected layers out of " \
                       f"{n_statistics_method} in a {model_type} model trained on {id_dataset}"
    else:
        file_title = f"{stat_name} gradient histogram"
        figure_title = f"gradients from 1 layer out of {len(n_statistics_plot)} in a {model_type} model"

    input_var_xlabel = get_input_var_xlabel(batch_size)

    xlabel = "$\\log f_{\\mathbf{\\theta}_\\ell}(" + input_var_xlabel +")  $"

    return file_title, figure_title, xlabel


def likelihoods_labels(model_type, model_name, batch_size, id_dataset, n_statistics_method, n_statistics_plot, single_figure, stat_name=None):

    file_title = f"likelihood histogram"
    figure_title = f"likelihoods for {model_type} model trained on {id_dataset}"

    input_var_xlabel = get_input_var_xlabel(batch_size)
    xlabel = "$ \\log p_{\\mathbf{\\theta}}(" + input_var_xlabel + ") $"

    return file_title, figure_title, xlabel



# TODO: refactor to nicely handle subclasses

label_getters = {
    "gradients_L2_norms": gradients_L2_norms_labels,
    "likelihoods": likelihoods_labels,
    "typicality": likelihoods_labels
}



def get_save_dir_path(model_name):
    save_dir_path = path.join(RUNNING_MODULE_DIR, model_name)
    if not path.exists(save_dir_path):
        makedirs(save_dir_path)

    return save_dir_path


def select_summary_stat_names(summary_stat_names, n_statistics):
    if n_statistics is None:
        return summary_stat_names
    else:
        # We preserve ordering for understandability
        stat_name_list = list(summary_stat_names)
        choices = random.choices(stat_name_list, k=n_statistics)
        return sorted(choices, key=(lambda name: stat_name_list.index(name)))


def fetch_preliminaries(model_type, model_name, model_mode, anomaly_detection_name, batch_size,
                        id_dataset, ood_dataset_names, fitted_distribution):
    anomaly_detection_method = anomaly_detection_methods_dict[anomaly_detection_name]

    id_dataset_summary, ood_dataset_summaries = get_dataset_summmaries(model_type, model_name, model_mode,
                                                                       anomaly_detection_method, batch_size,
                                                                       id_dataset, ood_dataset_names)

    anomaly_detector = anomaly_detection_method.from_dataset_summary(id_dataset_summary)

    if fitted_distribution:
        if not isinstance(anomaly_detector, DistributionFittingL2Norm):
            raise ValueError(f"anomaly detector is of type {type(anomaly_detector)} and thus the fitted "
                            f"distribution(s) can't be plotted.")

        # To fit the distributions, we actually need to run anomaly_detection_method.setup_method

        id_dataset_summary, fit_set_summary = anomaly_detector.split_dataset_summary(id_dataset_summary, 0.5)
        anomaly_detector.setup_method(fit_set_summary)

    return anomaly_detector, id_dataset_summary, ood_dataset_summaries


def plot_fitted_distribution(ax, anomaly_detector, stat_name):
    x_lim = ax.get_xlim()
    x = torch.linspace(*x_lim, steps=100)

    dist = anomaly_detector.fitted_log_scale_distribution(stat_name)
    p = torch.exp(dist.log_prob(x))
    ax.plot(x, p, color="b", label="fitted distribution")


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
        fig.suptitle(figure_title)

        plot_summary_histograms(ax, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names, stat_name, x_lim)

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

    save_dir_path = get_save_dir_path(model_name)

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
    fig.suptitle(figure_title)

    for stat_name, ax in zip(selected_stat_names, axs):

        plot_summary_histograms(ax, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names, stat_name, x_lim)

        if args.fitted_distribution:
            plot_fitted_distribution(ax, anomaly_detector, stat_name)

        ax.set_yticks([])
        ax.set_xlabel(xlabel) # "$ \\log f_{\\mathbf{\\theta}_\\ell}(\\mathbf{x}_{1:B})  $")

    # Grab the labels from the last axes to prevent label duplication
    fig.legend(*ax.get_legend_handles_labels())

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



