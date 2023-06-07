from path_definitions import PLOTS_DIR
from os import path
import sys

from models.utils import load_generative_model

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

OWN_DIR = path.join(PLOTS_DIR, "model_samples")

print("attempt 1: ", OWN_DIR)

OWN_DIR = path.realpath(sys.argv[0])

print("attempt 2:", OWN_DIR)

exit()

temp = 1

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
    plt.savefig(str(OWN_DIR) + title + ".png")

print("done")
