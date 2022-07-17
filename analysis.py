
from datasets import get_CIFAR10, get_SVHN

from model import Glow
import json

import numpy as np
import torch


device = torch.device("cuda")


svhn_path = "data/SVHN"
cifar_path = "data/CIFAR10"


_, _, _, vanilla_test_cifar = get_CIFAR10(False, "./", True)
_, _, _, vanilla_test_svhn = get_SVHN(False, "./", True)


def load_glow_model(output_folder, model_name, image_shape=(32, 32, 3), num_classes=10):
    with open(output_folder + 'hparams.json') as json_file:
        hparams = json.load(json_file)

    model = Glow(image_shape, hparams['hidden_channels'], hparams['K'], hparams['L'], hparams['actnorm_scale'],
                 hparams['flow_permutation'], hparams['flow_coupling'], hparams['LU_decomposed'], num_classes,
                 hparams['learn_top'], hparams['y_condition'])

    model.load_state_dict(torch.load(
        output_folder + model_name, map_location=device)["model"]) # You need to direct it "model" part of the file

    model.set_actnorm_init()

    model = model.to(device)

    model = model.eval()

    return model, hparams


def compute_nll(dataset, model, hparams):
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


def eval_img(img, model):
    latents, bpd, _ = model(img)
