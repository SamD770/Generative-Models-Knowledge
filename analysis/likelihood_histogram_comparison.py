
from analysis_utils import SampleDataset, load_generative_model, get_vanilla_dataset, compute_nll

import matplotlib.pyplot as plt

from torchvision.utils import make_grid

from torch.utils.data import Subset


model_names = ["cifar_long", "svhn_working",
               "imagenet32", "celeba"]

save_files = ["glow_checkpoint_585750.pt", "glow_checkpoint_280280.pt",
              "glow_checkpoint_400360.pt", "glow_checkpoint_419595.pt"]


dataset_names = ["cifar", "svhn", "imagenet32", "celeba"]


sample_dataset = True

draw_cross_entropy = True

save_dirs = [
    f"../glow_model/{model_name}/" for model_name in model_names
]

models = [
    load_generative_model("glow", save_dir, save_file) for save_dir, save_file in zip(save_dirs, save_files)
]


if sample_dataset:
    datasets = [
        SampleDataset(model) for model in models
    ]
else:
    datasets = [
        get_vanilla_dataset(dataset_name) for dataset_name in dataset_names
    ]


fig, axs = plt.subplots(nrows=4, ncols=2)

top_likelihood_ax, top_sample_ax = axs[0]

for model,model_name, ax in zip(models, model_names, axs):
# for model_name, save_file, ax in zip(model_names, save_files, axs):
#
#     save_dir = f"../glow_model/{model_name}/"
#     model = load_generative_model("glow", save_dir, save_file)

    likelihood_ax, sample_ax = ax

    likelihood_ax.sharex(top_likelihood_ax)

    likelihood_ax.set_xticks([])
    likelihood_ax.set_yticks([])

    likelihood_ax.set_ylabel(f"{model_name}")

    for dataset, dataset_name in zip(datasets, dataset_names):

        nlls = compute_nll(Subset(dataset, range(512*10)), model)
        nlls = nlls.clamp(max=11)

        print("-", end="")

        if likelihood_ax is top_likelihood_ax:
            label = dataset_name
        else:
            label=None

        likelihood_ax.hist(-nlls.numpy(), density=True, range=(-7, -1), bins=30, alpha=0.6, label=label)

        if draw_cross_entropy:
            cross_entropy = nlls.mean()
            likelihood_ax.vline(cross_entropy)     # should check this function.
        print("-")

    samples = model.generate_sample(32).cpu()
    samples = samples[:4]   # As currently glow_model can only generate samples of size 32
    sample_grid = make_grid(samples, nrow=2).permute(1, 2, 0)

    sample_ax.imshow(sample_grid)
    sample_ax.axis('off')

    print("#")

fig.legend()
fig.tight_layout()

plt.savefig("plots/seminal_paper_recreations/likelihood_histogram_comparison_samples.png")

print("done")


