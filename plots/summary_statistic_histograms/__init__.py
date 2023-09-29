import random
from os import path, makedirs

import numpy as np
import torch

from anomaly_methods.gradients.L2_norms import DistributionFittingL2Norm
from anomaly_methods.utils import anomaly_detection_methods_dict
from plots.utils import get_dataset_summmaries, RUNNING_MODULE_DIR, to_styled_dataset_name


random.seed(1)  # Fixes the seed so randomly selected layers are verifiable


def prepare_vals(summary, stat_name, take_log=True):
    if take_log:
        return prepare_vals_log(summary, stat_name)
    else:
        return prepare_vals_likelihoods(summary, stat_name)


def prepare_vals_log(summary, stat_name):
    return torch.log(summary[stat_name]).numpy()


def prepare_vals_likelihoods(summary, stat_name):
    return -summary[stat_name].numpy()


def plot_summary_histograms(ax, id_dataset_summary, id_dataset_name,
                            ood_dataset_summaries, ood_dataset_names, stat_name,
                            fit_id_x_lim=False, x_lim=None, take_log=True):
    """Plots the summary statistic stat_name as a histogram for the given summaries. providing x_lim overrides
    using fit_id_x_lim to fit to the in-distribution data."""

    # fit_id_x_lim tells us to scale the x_lim by the id_dataset
    if fit_id_x_lim and x_lim is None:
        id_vals = prepare_vals(id_dataset_summary, stat_name, take_log)
        x_lim = (id_vals.min(), id_vals.max())

    for dataset_name, summary in zip(ood_dataset_names, ood_dataset_summaries):

        styled_dataset_name = to_styled_dataset_name[dataset_name]

        if dataset_name == id_dataset_name:
            label=f"in distribution {styled_dataset_name}"
        else:
            label=f"out-of-distribution {styled_dataset_name}"

        vals = prepare_vals(summary, stat_name, take_log)
        ax.hist(vals, range=x_lim,
                label=label, density=True, bins=40, alpha=0.6)



def plot_summary_scatter(ax, id_dataset_summary, id_dataset_name,
                             ood_dataset_summaries, ood_dataset_names,
                             stat_name_x, stat_name_y, n_scatter=200):

    for dataset_name, summary in zip(ood_dataset_names, ood_dataset_summaries):
        if dataset_name == id_dataset_name:
            label=f"in distribution {id_dataset_name}"
        else:
            label=f"out-of-distribution {dataset_name}"

        x_vals = prepare_vals(summary, stat_name_x)[:n_scatter]
        y_vals = prepare_vals(summary, stat_name_y)[:n_scatter]

        ax.scatter(x_vals, y_vals, marker=".", label=label)


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
    "gradients_L2_norms_gaussian": gradients_L2_norms_labels,
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


def plot_fitted_distribution(ax, anomaly_detector, stat_name, color="k"):
    x_lim = ax.get_xlim()
    x = torch.linspace(*x_lim, steps=100)

    dist = anomaly_detector.fitted_log_scale_distribution(stat_name)
    p = torch.exp(dist.log_prob(x))
    ax.plot(x, p, color=color, label="fitted distribution")


def plot_fitted_distribution_scatter(ax, anomaly_detector, stat_name_x, stat_name_y, color="k"):
    x_lim = ax.get_xlim()
    x = torch.linspace(*x_lim, steps=100)

    y_lim = ax.get_ylim()
    y = torch.linspace(*y_lim, steps=100)

    dist_x = anomaly_detector.fitted_log_scale_distribution(stat_name_x)

    log_p_x = dist_x.log_prob(x)

    dist_y = anomaly_detector.fitted_log_scale_distribution(stat_name_y)
    log_p_y = dist_y.log_prob(y)

    # Take cartesian products to get them independently
    X, Y = np.meshgrid(x, y)
    log_p_X, log_p_Y = np.meshgrid(log_p_x, log_p_y)

    p_joint = np.exp(log_p_X + log_p_Y)

    ax.contour(X, Y, p_joint, colors=color)
