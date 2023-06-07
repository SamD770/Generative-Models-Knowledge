from path_definitions import PLOTS_DIR
from os import path

from models.utils import load_generative_model

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

OWN_DIR = path.join(PLOTS_DIR, "sampling_plots")

temp = 1

file_list = ["VAE_cifar.pt"]

for file in file_list:
    print("sampling from", file)

    model = load_generative_model("vae", file, input_shape=(32, 32, 3), latent_dims=64)

    model.to("cuda")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("params:", pytorch_total_params)

    samples = model.generate_sample(32).cpu()
    print(f"range {(samples.min(), samples.max())}")

    title = f"samples from {file} model"
    grid = make_grid(samples, nrow=8).permute(1, 2, 0) + 0.5

    # plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    plt.savefig(str(OWN_DIR) + title + ".png")

print("done")
