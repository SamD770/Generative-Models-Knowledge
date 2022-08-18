from analysis.analysis_utils import load_glow_model

from pixelCNN_model.main import PixelCNN

from data.datasets import postprocess
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

temp = 1


model_name_list = ["FashionMNIST PixelCNN"]
file_list = ["PixelCNN_FashionMNIST_checkpoint.pt"]

for model_name, file in zip(model_name_list, file_list):

    print("sampling from", model_name)
    # model_dir = f"../glow_model/{model_name}/"

    model_dir = "../pixelCNN_model/"

    model = PixelCNN.load_serialised(model_dir, file)
    samples = model.generate_sample(32).cpu()

    # model = load_glow_model(model_dir, file)
    # samples = postprocess(model(temperature=1, reverse=True)).cpu()

    title = f"samples from {model_name} model"
    grid = make_grid(samples, nrow=8).permute(1, 2, 0)
    # plt.title(title)
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig("plots/sample_plots/" + title + ".png")

print("done")