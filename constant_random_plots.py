import json
import pywt

import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from datasets import get_CIFAR10, get_SVHN
from model import Glow


class RandomNoiseDataset:
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = image_shape
        print(f"image_shape: {self.image_shape}")

    def __len__(self):
        return 512

    def __getitem__(self, item):
        means = torch.zeros(self.image_shape)
        stds = torch.ones(self.image_shape)/5
        return torch.normal(means, stds), torch.zeros(10)


class ConstantDataset:
    def __init__(self, image_shape):
        super().__init__()
        self.image_shape = image_shape

    def __len__(self):
        return 512

    def __getitem__(self, item):
        return item*torch.ones(self.image_shape)/10000, torch.zeros(10)


device = torch.device("cuda")

output_folder = "cifar_glow/"
model_name = 'glow_checkpoint_194469.pt'

with open(output_folder + 'hparams.json') as json_file:
    hparams = json.load(json_file)

print(hparams)
print(f"using model: {output_folder + model_name}")


image_shape, num_classes, _, test_cifar = get_CIFAR10(hparams['augment'], hparams['dataroot'], True)
image_shape, num_classes, _, test_svhn = get_SVHN(hparams['augment'], hparams['dataroot'], True)

random_data = RandomNoiseDataset((3, 32, 32))
constant_data = ConstantDataset((3, 32, 32))


model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
             hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
             hparams['learn_top'], hparams['y_condition'])


model.load_state_dict(torch.load(
    output_folder + model_name, map_location=device)["model"]) # You need to direct it "model" part of the file


model.set_actnorm_init()

model = model.to(device)

model = model.eval()


def compute_nll(dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=6)

    nlls = []
    for x, y in dataloader:
        x = x.to(device)

        if hparams['y_condition']:
            y = y.to(device)
        else:
            y = None

        with torch.no_grad():
            _, nll, _ = model(x, y_onehot=y)
            nlls.append(nll)

    return torch.cat(nlls).cpu()


# random_nll = compute_nll(random_data, model)
# constant_nll = compute_nll(constant_data, model)
# cifar_nll = compute_nll(test_cifar, model)
# svhn_nll = compute_nll(test_svhn, model)
#
# print("CIFAR NLL", torch.mean(cifar_nll))
# print("SVHN NLL", torch.mean(svhn_nll))

plt.figure(figsize=(20,10))
plt.title("Histogram Glow - trained on CIFAR10")
plt.xlabel("Negative bits per dimension")
# plt.hist(-svhn_nll.numpy(), label="SVHN", density=True, alpha=0.6, bins=30)
# plt.hist(-cifar_nll.numpy(), label="CIFAR10", density=True, alpha=0.6, bins=50)


for dataset, name in zip(
        [random_data, constant_data, test_cifar, test_svhn],
        ["gaussian noise", "constant", "CIFAR10", "SVHN"]
    ):
    nll = compute_nll(dataset, model)
    print(f"{name} NLL:", torch.mean(nll))
    plt.hist(-nll.numpy(), label=name, density=True, alpha=0.6, bins=30)

plt.legend()
# plt.show()
plt.savefig("images/glow_nll_with_constant_random_2.png", dpi=300)