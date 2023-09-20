
from os import path

import matplotlib.pyplot as plt

from plots.summary_statistic_histograms import plot_summary_histograms, plot_summary_scatter, fetch_preliminaries, \
    get_save_dir_path, select_summary_stat_names, plot_fitted_distribution, label_getters


def run(model_type, model_name, model_mode, anomaly_detection_name, batch_size, id_dataset, ood_dataset_names,
        fitted_distribution, x_lim):
    """Plots axes (one for each summary statistic) on one figure."""

    # Fetch cached statistics from the disk

    anomaly_detector, id_dataset_summary, ood_dataset_summaries = \
        fetch_preliminaries(model_type, model_name, model_mode, anomaly_detection_name, batch_size,
                            id_dataset, ood_dataset_names, fitted_distribution)

    save_dir_path = get_save_dir_path(model_name)

    # plot histograms of the data

    selected_stat_names = select_summary_stat_names(anomaly_detector.summary_statistic_names, 2)
    fig, axs = plt.subplots(ncols=3)

    label_getter = label_getters[anomaly_detection_name]

    file_title, figure_title, xlabel = label_getter(
        model_type, model_name, batch_size, id_dataset, anomaly_detector.summary_statistic_names, 2,
        single_figure=True, stat_name=None
    )

    # file_title = f"{model_type} {model_name} gradient histogram"

    filepath = path.join(save_dir_path, file_title + ".png")

    with open(filepath[:-4] + ".txt", "wt") as f:  # quick and dirty way to record the names used
        f.write(str(selected_stat_names))

    print(f"creating: {filepath}")

    fig.suptitle(figure_title)

    histogram_axs = axs[:-1]
    scatter_ax = axs[-1]

    for stat_name, ax in zip(selected_stat_names, histogram_axs):

        plot_summary_histograms(ax, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names, stat_name,
                                x_lim)

        if fitted_distribution:
            plot_fitted_distribution(ax, anomaly_detector, stat_name)

        ax.set_yticks([])
        ax.set_xlabel(xlabel)

    plot_summary_scatter(scatter_ax, id_dataset_summary, id_dataset, ood_dataset_summaries, ood_dataset_names,
                         selected_stat_names[0], selected_stat_names[1])

    # Grab the labels from the last axes to prevent label duplication
    fig.legend(*scatter_ax.get_legend_handles_labels())

    plt.savefig(filepath)

