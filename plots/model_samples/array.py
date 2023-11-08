"""For plotting arrays of samples from different models."""
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


def run(model_type, model_names):
    nrow = 8

    samples = []
    for model_name in model_names:
        model = load_generative_model(model_type, model_name)
        model.to(device)

        samples.append(
            model.generate_sample(32).cpu()[:8]
        )

    title = f"samples from {model_type}"

    samples = torch.cat(samples, 0)

    grid = grid_from_imgs(samples)

    # plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    save_plot(title)

    print("done")


parser = argparse.ArgumentParser(parents=[model_parser])

args = parser.parse_args()

run(args.model_type, args.model_names)