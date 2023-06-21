from path_definitions import PLOTS_DIR
from os import path
import sys

from plots.utils import RUNNING_MODULE_DIR

from models.utils import load_generative_model

from torchvision.utils import make_grid

import matplotlib.pyplot as plt


def run(model):
    pass


file_list = ["VAE_cifar.pt"]

name_list = ["cifar_glow"]

for name in name_list:
    print("sampling from", name)

    model = load_generative_model("glow", name)

    model.to("cuda")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("params:", pytorch_total_params)

    samples = model.generate_sample(32).cpu()
    print(f"range {(samples.min(), samples.max())}")

    title = f"samples from {name} model"
    grid = make_grid(samples, nrow=8).permute(1, 2, 0) + 0.5

    # plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    save_dir = path.join(RUNNING_MODULE_DIR, f"({title}).png")

print("done")
