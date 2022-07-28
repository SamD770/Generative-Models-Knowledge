
from datasets import get_CIFAR10, get_SVHN

from glow_model.model import Glow
import json

import torch


def vanilla_test_cifar():
    _, _, _, test_cifar = get_CIFAR10(False, "./", True)
    return test_cifar


def vanilla_test_svhn():
    _, _, _, test_svhn = get_SVHN(False, "./", True)
    return test_svhn


class SampleDataset:
    def __init__(self, model, batch_count=128, temp=1):
        """batch_count is the number of 32-length batches to generate"""
        super().__init__()
        self.batch_count = batch_count
        self.samples = []

        for _ in range(self.batch_count):

            imgs = model(temperature=temp, reverse=True).cpu()

            for img in imgs:
                self.samples.append(img)

    def __len__(self):
        return self.batch_count * 32

    def __getitem__(self, item):
        return self.samples[item], torch.zeros(10)


device = torch.device("cpu")


svhn_path = "../data/SVHN"
cifar_path = "../data/CIFAR10"


# _, _, _, vanilla_test_cifar = get_CIFAR10(False, "./", True)
# _, _, _, vanilla_test_svhn = get_SVHN(False, "./", True)


def load_glow_model(output_folder, model_name, image_shape=(32, 32, 3), num_classes=10):
    with open(output_folder + 'hparams.json') as json_file:
        hparams = json.load(json_file)

    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                 hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                 hparams['learn_top'], hparams['y_condition'])


    print(output_folder + model_name)
    model.load_state_dict(torch.load(
        output_folder + model_name, map_location=device)["model"]) # You need to direct it "model" part of the file

    model.set_actnorm_init()

    model = model.to(device)

    model = model.eval()

    return model, hparams


def compute_nll(dataset, model, hparams):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=1)

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


