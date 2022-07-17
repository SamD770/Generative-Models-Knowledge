import json
import pywt

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from datasets import get_CIFAR10, get_SVHN, preprocess, one_hot_encode
from model import Glow


from torchvision import transforms, datasets

path = "data/SVHN"

SVHN_test = datasets.SVHN(
    path,
    split="test",
    transform=None,
    target_transform=None,
    download=True,
)


def wavelet_strip(channel, wavelet="bior1.3", low_scale=0):
    """Takes out the bottom three quadrants of the wavelet transform of the image."""
    cA, (cH, cV, cD) = pywt.dwt2(channel, wavelet)
    high_level = pywt.idwt2((cA, (cH*low_scale, cV*low_scale, cD*low_scale)), wavelet)
    return high_level


def image_wavelet_treatment(img, wavelet="bior1.3", low_scale=0):
    """Takes in a PIL image and outputs a numpy array that can be passed to ToTensor()."""
    img_arr = np.array(img)
    channels = [img_arr[:, :, i] for i in range(3)]
    high_level = [
        wavelet_strip(chan, wavelet=wavelet, low_scale=low_scale) for chan in channels
    ]
    high_level = np.stack(high_level, 2)
    high_level = high_level/255
    high_level = high_level.astype(float)
    return high_level


img, _ = SVHN_test[3]

test_transform = transforms.Compose([transforms.ToTensor(), preprocess])
wavelet_preprocess = transforms.Compose([image_wavelet_treatment, transforms.ToTensor(), preprocess])

# print(test_transform(img))
#
# print("\n" * 5)
#
# print(wavelet_preprocess(img))


wavelet_svhn = datasets.SVHN(
    path,
    split="test",
    transform=wavelet_preprocess,
    target_transform=one_hot_encode,
    download=True,
)

normal_svhn =  datasets.SVHN(
    path,
    split="test",
    transform=test_transform,
    target_transform=one_hot_encode,
    download=True,
)


wavelet_cifar = datasets.CIFAR10(
    path,
    train=False,
    transform=wavelet_preprocess,
    target_transform=one_hot_encode,
    download=True,
)

normal_cifar = datasets.CIFAR10(
    path,
    train=False,
    transform=test_transform,
    target_transform=one_hot_encode,
    download=True,
)


device = torch.device("cuda")


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
        x = x.float()

        if hparams['y_condition']:
            y = y.to(device)
        else:
            y = None

        with torch.no_grad():
            _, nll, _ = model(x, y_onehot=y)
            nlls.append(nll)

    return torch.cat(nlls).cpu()


output_folder = "output_new/"
model_name = 'glow_checkpoint_194469.pt'


def make_likelihood_histogram(datasets, names, plot_file_name):

    model, hparams = load_glow_model(output_folder, model_name)
    plt.figure(figsize=(20, 10))
    plt.title("Histogram Glow - trained on CIFAR10")
    plt.xlabel("Negative bits per dimension")

    for dataset, name in zip(datasets, names):
        nll = compute_nll(dataset, model, hparams)
        print(f"{name} NLL:", torch.mean(nll))
        plt.hist(-nll.numpy(), label=name, density=True, alpha=0.6, bins="rice")

    plt.legend()
    plt.savefig(plot_file_name, dpi=300)


make_likelihood_histogram([wavelet_cifar, wavelet_svhn],
                          ["wavelet cifar", "wavelet svhn"],
                          "images/glow_nll_only_wavelets.png")

#
# high_level = np.stack(high_level, -1)
#
# mask = np.ones((16, 16))
# mask = np.pad(mask, ((0, 16), (0, 16)), 'constant', constant_values=((0, 0), (0, 0)))
#
#
# r_fourier_coeffs = np.fft.fft2(r)
#
# r_recon = np.fft.ifft2(mask*r_fourier_coeffs)
#
#
# print(np.fft.rfft2(r))
#
# print(r_recon)
#
#
# fig, (ax1, ax2) = plt.subplots(2)
#
# ax1.imshow(high_level)
#
# ax2.imshow(img_arr)
#
# plt.show()
#
