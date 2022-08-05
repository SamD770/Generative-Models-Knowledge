from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision import transforms, datasets

n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x


def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()


def one_hot_encode(target):
    """
    One hot encode with fixed 10 classes
    Args: target           - the target labels to one-hot encode
    Retn: one_hot_encoding - the OHE of this tensor
    """
    num_classes = 10
    one_hot_encoding = F.one_hot(torch.tensor(target),num_classes)

    return one_hot_encoding


def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    path = Path(dataroot) / "data" / "CIFAR10"
    train_dataset = datasets.CIFAR10(
        path,
        train=True,
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.CIFAR10(
        path,
        train=False,
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset


def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    train_transform = transforms.Compose(transformations)

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    path = Path(dataroot) / "data" / "SVHN"
    train_dataset = datasets.SVHN(
        path,
        split="train",
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.SVHN(
        path,
        split="test",
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset



def get_imagenet32(dataroot):
    image_shape = (32, 32, 3)
    num_classes = None  # TODO @Sam

    X_train_list = []
    for i in range(1, 11):
        X_train_batch = load_databatch(os.path.join(dataroot, "imagenet32_regular", 'train_32x32', 'train_data_batch_' + str(i)))
        X_train_list.append(X_train_batch)
    X_train = np.concatenate(X_train_list)
    X_test = load_databatch(os.path.join(dataroot, "imagenet32_regular", "valid_32x32", "val_data"))

    # TODO @Sam: you may want to replace this by another Dataset class. Not that TensorDataset will return a list of tensors for every batch, which contains exactly one element.
    train_dataset = torch.utils.data.TensorDataset(torch.as_tensor(X_train, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.as_tensor(X_test, dtype=torch.float32))

    return image_shape, num_classes, train_dataset, test_dataset
def get_celeba(dataroot):
    image_shape = (32, 32, 3)
    resize = 32

    num_classes = None  # TODO @Sam

    train_transform, valid_transform = _data_transforms_celeba64(resize)
    train_data = LMDBDataset(root='data/celeba64_lmdb', name='celeba64', split="train", transform=train_transform,
                             is_encoded=True)
    valid_data = LMDBDataset(root='data/celeba64_lmdb', name='celeba64', split="validation", transform=valid_transform,
                             is_encoded=True)
    test_data = LMDBDataset(root='data/celeba64_lmdb', name='celeba64', split="test", transform=valid_transform,
                            is_encoded=True)

    X_train = np.zeros((len(train_data), resize, resize, 3))
    for i in range(len(train_data)):
        X_train[i, :, :, :] = np.einsum('ijk -> jki', train_data[i])

    X_val = np.zeros((len(valid_data), resize, resize, 3))
    for i in range(len(valid_data)):
        X_val[i, :, :, :] = np.einsum('ijk -> jki', valid_data[i])

    X_test = np.zeros((len(test_data), resize, resize, 3))
    for i in range(len(test_data)):
        X_test[i, :, :, :] = np.einsum('ijk -> jki', test_data[i])

    # scale to [0, 255] as expected
    X_train = X_train * 255.
    X_val = X_val * 255.
    X_test = X_test * 255.

    # round to nearest integer
    X_train = np.round(X_train, decimals=0)
    X_val = np.round(X_val, decimals=0)
    X_test = np.round(X_test, decimals=0)

    # convert to uint8
    X_train = X_train.astype('uint8')
    X_val = X_val.astype('uint8')
    X_test = X_test.astype('uint8')



    # print(trX, vaX, teX)
    # print("celeba64 A: ", np.min(X_train.flatten()), np.max(X_train.flatten()), np.min(X_val.flatten()), np.max(X_val.flatten()), np.min(X_test.flatten()), np.max(X_test.flatten()))
    # print("celeba64 B: ", X_train.dtype, X_val.dtype, X_test.dtype)

    # TODO @Sam: you may want to replace this by another Dataset class. Not that TensorDataset will return a list of tensors for every batch, which contains exactly one element.
    train_dataset = torch.utils.data.TensorDataset(torch.as_tensor(X_train, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.as_tensor(X_test, dtype=torch.float32))

    return image_shape, num_classes, train_dataset, test_dataset




# --------------



def flatten(outer):
    return [el for inner in outer for el in inner]


def load_databatch(path, img_size=32):
    """
    As copied from https://patrykchrabaszcz.github.io/Imagenet32/
    """
    d = unpickle_imagenet32(path)
    x = d['data']  # is already uint8

    img_size2 = img_size * img_size
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))   # .transpose(0, 3, 1, 2) (do not transpose, since we want (samples, 32, 32, 3))

    return x

def unpickle_imagenet32(file):
    """
    As copied from https://patrykchrabaszcz.github.io/Imagenet32/
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        # transforms.RandomHorizontalFlip(),  # taken out compared to NVAE --> we don't want data augmentation
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform

class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image


def num_samples(dataset, split):
    if dataset == 'celeba':
        # return 27000 if train else 3000
        pass
    elif dataset == 'celeba64':
        if split == "train":
            return 162770
        elif split == "validation":
            return 19867
        elif split == "test":
            return 19962
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


class LMDBDataset(data.Dataset):
    def __init__(self, root, name='', split="train", transform=None, is_encoded=False):
        self.name = name
        self.split = split
        self.transform = transform
        if self.split in ["train", "validation", "test"]:
            lmdb_path = os.path.join(root, f'{self.split}.lmdb')
        else:
            print('invalid split param')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB')
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return num_samples(self.name, self.split)