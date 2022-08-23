
from analysis_utils import SampleDataset, load_generative_model, get_vanilla_test_dataset, compute_nll

import matplotlib.pyplot as plt

from torchvision.utils import make_grid


model_names = []
save_files = []


dataset_names = []


sample_dataset = False

draw_cross_entropy = False

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
        get_vanilla_test_dataset(dataset_name) for dataset_name in dataset_names
    ]


fig, axs = plt.subplot(nrows=4, ncols=2)


for model_name, save_file, ax in zip(model_names, save_files, axs):

    save_dir = f"../glow_model/{model_name}/"
    model = load_generative_model("glow", save_dir, save_file)

    likelihood_ax, sample_ax = ax

    for dataset in datasets:
        nlls = compute_nll(dataset, model)

        likelihood_ax.hist(-nlls, density=True)

        cross_entropy = nlls.mean()

        if draw_cross_entropy:
            likelihood_ax.x_vert(cross_entropy)     # should check this function.

    samples = model.generate_sample(32).cpu()
    samples = samples[:4]   # As currently glow_model can only generate samples of size 32
    sample_grid = make_grid(samples, nrow=2)

    sample_ax.imshow(sample_grid)
    sample_ax.axis('off')

fig.legend()

plt.savefig("plots/seminal_paper_recreations/likelihood_histogram_comparison.png")


