from path_definitions import PLOTS_DIR
from os import path
import sys
import torch

from plots.utils import save_plot, grid_from_imgs
from command_line_utils import model_parser

from models.utils import load_generative_model

import matplotlib.pyplot as plt

import argparse

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)


def run(model_type, model_name):
    model = load_generative_model(model_type, model_name)
    model.to(device)
    #
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print("params:", pytorch_total_params)

    samples = model.generate_sample(32).cpu()
    print(f"range {(samples.min(), samples.max())}")

    title = f"samples from {name} model"
    grid = grid_from_imgs(samples)

    # plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    save_plot(title)


    print("done")

"""
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
"""

parser = argparse.ArgumentParser(parents=[model_parser])

args = parser.parse_args()

for name in args.model_names:
    run(args.model_type, name)