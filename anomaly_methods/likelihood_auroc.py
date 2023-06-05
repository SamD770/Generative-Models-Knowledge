from data.utils import (
    SampleDataset,
    get_vanilla_dataset,
)
from models.utils import load_generative_model, compute_nll

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

from torch.utils.data import Subset

import torch


model_names = ["cifar_long", "svhn_working", "imagenet32", "celeba"]

save_files = [
    "glow_checkpoint_585750.pt",
    "glow_checkpoint_280280.pt",
    "glow_checkpoint_400360.pt",
    "glow_checkpoint_419595.pt",
]


dataset_names = ["cifar", "svhn", "imagenet32", "celeba"]


sample_dataset = False

draw_cross_entropy = False

save_dirs = [f"./glow_model/{model_name}/" for model_name in model_names]

models = [
    load_generative_model("glow", save_file, save_dir)
    for save_dir, save_file in zip(save_dirs, save_files)
]


if sample_dataset:
    datasets = [SampleDataset(model) for model in models]
else:
    datasets = [get_vanilla_dataset(dataset_name) for dataset_name in dataset_names]


fig, axs = plt.subplots(nrows=4, ncols=1)

top_likelihood_ax = axs[0]  # , top_sample_ax = axs[0]
last_likelihood_ax = axs[-1]


def fit_roc_curve(id_scores, ood_scores):
    y_true = torch.cat([torch.ones(len(id_scores)), torch.zeros(len(ood_scores))])

    y_scores = torch.cat([torch.tensor(id_scores), torch.tensor(ood_scores)])

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    return fpr, tpr, auc(fpr, tpr)


for model, model_name, ax in zip(models, dataset_names, axs):
    id_dataset_name = None
    id_nll = None

    ood_dataset_names = []
    ood_dataset_nlls = []

    for dataset, dataset_name in zip(datasets, dataset_names):
        nlls = compute_nll(Subset(dataset, range(512 * 10)), model)

        print(nlls.shape)

        if dataset_name == model_name:
            id_dataset_name = dataset_name
            id_nll = nlls
        else:
            ood_dataset_names.append(dataset_name)
            ood_dataset_nlls.append(nlls)

    print("in distribution: ", id_dataset_name)
    id_ll = -id_nll.clone().detach().numpy()
    for ood_dataset_name, ood_nll in zip(ood_dataset_names, ood_dataset_nlls):
        ood_ll = -ood_nll.clone().detach().numpy()
        _, _, area = fit_roc_curve(id_ll, ood_ll)
        print()
        print(f"ood dataset {ood_dataset_name} has auroc {area}")

    print()
    print()
