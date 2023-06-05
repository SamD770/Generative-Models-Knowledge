import json

import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from data.datasets import get_CIFAR10, get_SVHN
from models.glow_model import Glow

device = torch.device("cuda")

output_folder = "cifar_glow/"
model_name = "glow_checkpoint_194469.pt"

with open(output_folder + "hparams.json") as json_file:
    hparams = json.load(json_file)

print(hparams)
print(f"using model: {output_folder + model_name}")


image_shape, num_classes, _, test_cifar = get_CIFAR10(
    hparams["augment"], hparams["dataroot"], True
)
image_shape, num_classes, _, test_svhn = get_SVHN(
    hparams["augment"], hparams["dataroot"], True
)


model = Glow(
    image_shape,
    hparams["hidden_channels"],
    hparams["K"],
    hparams["L"],
    hparams["actnorm_scale"],
    hparams["flow_permutation"],
    hparams["flow_coupling"],
    hparams["LU_decomposed"],
    num_classes,
    hparams["learn_top"],
    hparams["y_condition"],
)


model.load_state_dict(torch.load(output_folder + model_name)["model"])
# You need to direct it to the "model" part of the file

model.set_actnorm_init()

model = model.to(device)

model = model.eval()


def compute_nll(dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=6)

    nlls = []
    for x, y in dataloader:
        x = x.to(device)

        if hparams["y_condition"]:
            y = y.to(device)
        else:
            y = None

        with torch.no_grad():
            _, nll, _ = model(x, y_onehot=y)
            nlls.append(nll)

    return torch.cat(nlls).cpu()


cifar_nll = compute_nll(test_cifar, model)
svhn_nll = compute_nll(test_svhn, model)

print("CIFAR NLL", torch.mean(cifar_nll))
print("SVHN NLL", torch.mean(svhn_nll))

plt.figure(figsize=(20, 10))
plt.title("Histogram Glow - trained on CIFAR10")
plt.xlabel("Negative bits per dimension")
plt.hist(-svhn_nll.numpy(), label="SVHN", density=True, alpha=0.6, bins=30)
plt.hist(-cifar_nll.numpy(), label="CIFAR10", density=True, alpha=0.6, bins=50)
plt.legend()
# plt.show()
plt.savefig("new_histogram_glow_cifar_svhn.png", dpi=300)
