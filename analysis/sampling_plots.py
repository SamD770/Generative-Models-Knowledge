from analysis import *

from data.datasets import postprocess
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

temp = 1

svhn_model, svhn_hparams = load_glow_model("svhn_glow/", "glow_checkpoint_286000.pt")
cifar_model, cifar_hparams = load_glow_model("cifar_glow/", "glow_checkpoint_195250.pt")


svhn_images = postprocess(svhn_model(temperature=1, reverse=True)).cpu()
cifar_images = postprocess(cifar_model(temperature=1, reverse=True)).cpu()


grid = make_grid(cifar_images, nrow=8).permute(1,2,0)

# plt.figure(figsize=(10,10))
plt.imshow(grid)
plt.axis('off')

plt.savefig("plots/cifar_samples.png", dpi=300)
