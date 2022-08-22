
from data.datasets import get_CIFAR10, get_SVHN, get_celeba, get_imagenet32, get_MNIST, get_FashionMNIST

from glow_model.model import Glow
from pixelCNN_model.main import PixelCNN

import json

import torch
from torch.utils.data import IterableDataset


device = torch.device("cuda")
print(f"using device: {device}")


svhn_path = "../data/SVHN"
cifar_path = "../data/CIFAR10"


def vanilla_test_cifar():
    _, _, _, test_cifar = get_CIFAR10(False, "../", True)
    return test_cifar


def vanilla_test_svhn():
    _, _, _, test_svhn = get_SVHN(False, "../", True)
    return test_svhn


def vanilla_test_celeba():
    _, _, _, test_celeba = get_celeba("../")
    return test_celeba


def vanilla_test_imagenet32():
    _, _, _, test_imagenet32 = get_imagenet32("../")
    return test_imagenet32


def vanilla_test_FashionMNIST():
    _, _, _, test_FashionMNIST = get_FashionMNIST("../")
    return test_FashionMNIST


def vanilla_test_MNIST():
    _, _, _, test_MNIST = get_MNIST("../")
    return test_MNIST


def get_vanilla_test_dataset(dataset_name):
    return {
        "cifar": vanilla_test_cifar,
        "svhn": vanilla_test_svhn,
        "celeba": vanilla_test_celeba,
        "imagenet32": vanilla_test_imagenet32,
        "FashionMNIST": vanilla_test_FashionMNIST,
        "MNIST": vanilla_test_MNIST
    }[dataset_name]()


def load_generative_model(model_type, save_dir, save_file, **params):
    return {
        "glow": Glow,
        "PixelCNN": PixelCNN
    }[model_type].load_serialised(save_dir, save_file, **params)


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



def load_glow_model(output_folder, model_name, image_shape=(32, 32, 3), num_classes=10):

    print(f"loading model from: {output_folder + model_name}")

    with open(output_folder + 'hparams.json') as json_file:
        hparams = json.load(json_file)

    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                 hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                 hparams['learn_top'], hparams['y_condition'])

    state_dicts = torch.load(
        output_folder + model_name, map_location=device)
    print(f"stored information: {state_dicts.keys()}")

    model.load_state_dict(state_dicts["model"]) # You need to direct it "model" part of the file

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
            nll = model.eval_nll(x)
            nlls.append(nll)

    return torch.cat(nlls).cpu()





