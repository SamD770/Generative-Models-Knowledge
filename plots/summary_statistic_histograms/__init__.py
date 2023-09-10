import torch


def plot_summary_histograms(ax, id_dataset_summary, id_dataset_name,
                            ood_dataset_summaries, ood_dataset_names, stat_name, x_lim=None):

    if x_lim is None:
        id_vals = torch.log(id_dataset_summary[stat_name]).numpy()
        x_lim = (id_vals.min(), id_vals.max())

    for dataset_name, summary in zip(ood_dataset_names, ood_dataset_summaries):

        if dataset_name == id_dataset_name:
            label=f"in distribution {id_dataset_name}"
        else:
            label=f"out-of-distribution {dataset_name}"

        vals = torch.log(summary[stat_name]).numpy()
        ax.hist(vals, range=x_lim,
                label=label, density=True, bins=40, alpha=0.6)
