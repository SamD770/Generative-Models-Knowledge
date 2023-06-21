"""
Defines the Datasets used in the
"""

import warnings

from pathlib import Path
from os import path

import torch
import torch.nn.functional as F
import pickle

from torchvision import transforms, datasets
from path_definitions import DATAROOT


import torch.utils.data as data
import numpy as np
import lmdb
import io
from PIL import Image

# All greyscale datasets are scaled from [0, 255] to [0, 1]
# All color datasets are scaled from [0, 255] to [-0.5, 0.5]


class DatasetWrapper:
    """
    Primarily associates a fixed name with each dataset so that it can be used in the command line.
    """
    name = NotImplementedError()
    image_shape = NotImplementedError()
    num_classes = NotImplementedError()

    @staticmethod
    def get_train(dataroot=DATAROOT):
        raise NotImplementedError()

    @staticmethod
    def get_test(dataroot=DATAROOT):
        raise NotImplementedError()

    @classmethod
    def get_all(cls, dataroot=DATAROOT):
        """Returns a tuple of data used for the dataset (for backwards compatibility with Glow code)."""
        return cls.image_shape, cls.num_classes, cls.get_train(dataroot), cls.get_test(dataroot)


def MNIST_scaling(x):
    return x - 0.5


class MNIST_Wrapper(DatasetWrapper):

    name = "MNIST"
    image_shape = (28, 28, 1)
    num_classes = 10

    @staticmethod
    def get_train(dataroot=DATAROOT):
        return datasets.MNIST(
            dataroot, train=True, download=True, transform=transforms.ToTensor()
        )

    @staticmethod
    def get_test(dataroot=DATAROOT):
        return datasets.MNIST(
            dataroot, train=False, download=True, transform=transforms.ToTensor()
        )   
    

def get_MNIST(dataroot):
    warnings.warn("DEPRECATED. Use MNIST_Wrapper.get_all instead.", DeprecationWarning)
    image_shape = (28, 28, 1)

    num_classes = 10

    train_dataset = datasets.MNIST(
        DATAROOT, train=True, download=True, transform=transforms.ToTensor()
    )
    # transform=transforms.Compose([transforms.ToTensor(), MNIST_scaling]))

    test_dataset = datasets.MNIST(
        DATAROOT, train=False, download=True, transform=transforms.ToTensor()
    )
    # transform=transforms.Compose([transforms.ToTensor(), MNIST_scaling]))

    return image_shape, num_classes, train_dataset, test_dataset


class FashionMNIST_Wrapper(DatasetWrapper):
    name = "FashionMNIST"
    image_shape = (28, 28, 1)
    num_classes = 10

    @staticmethod
    def get_train(dataroot=DATAROOT):
        return datasets.FashionMNIST(
            dataroot, train=True, download=True, transform=transforms.ToTensor()
        )

    @staticmethod
    def get_test(dataroot=DATAROOT):
        return datasets.FashionMNIST(
            dataroot, train=False, download=True, transform=transforms.ToTensor()
        )


def get_FashionMNIST(dataroot):
    
    warnings.warn("DEPRECATED. Use FashionMNIST_Wrapper.get_all instead.", DeprecationWarning)
    
    image_shape = (28, 28, 1)

    num_classes = 10

    train_dataset = datasets.FashionMNIST(
        DATAROOT, train=True, download=True, transform=transforms.ToTensor()
    )
    # transform=transforms.Compose([transforms.ToTensor(), MNIST_scaling]))

    test_dataset = datasets.FashionMNIST(
        DATAROOT, train=False, download=True, transform=transforms.ToTensor()
    )
    # transform=transforms.Compose([transforms.ToTensor(), MNIST_scaling]))

    return image_shape, num_classes, train_dataset, test_dataset


class Omniglot_Wrapper(DatasetWrapper):
    name = "Omniglot"
    image_shape = (28, 28, 1)
    num_classes = 10

    scaling_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((image_shape[0], image_shape[1]))]
    )

    @staticmethod
    def get_train(dataroot=DATAROOT):
        return datasets.Omniglot(
            dataroot, background=True, download=True, transform=Omniglot_Wrapper.scaling_transform
        )

    @staticmethod
    def get_test(dataroot=DATAROOT):
        return datasets.Omniglot(
            dataroot, background=False, download=True, transform=Omniglot_Wrapper.scaling_transform
        )


def flip(x):
    return 1 - x


class FlippedOmniglotWrapper(Omniglot_Wrapper):
    name = "flipped_Omniglot"

    scaling_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            flip,
            transforms.Resize((Omniglot_Wrapper.image_shape[0], Omniglot_Wrapper.image_shape[1])),
        ]
    )


def get_Omniglot(dataroot):
    warnings.warn("DEPRECATED. Use Omniglot_Wrapper.get_all instead.", DeprecationWarning)

    image_shape = (28, 28, 1)

    num_classes = 10

    scaling_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((image_shape[0], image_shape[1]))]
    )

    train_dataset = datasets.Omniglot(
        DATAROOT, background=True, download=True, transform=scaling_transform
    )

    test_dataset = datasets.Omniglot(
        DATAROOT, background=False, download=True, transform=scaling_transform
    )

    return image_shape, num_classes, train_dataset, test_dataset


def get_flipped_Omniglot(dataroot):
    warnings.warn("DEPRECATED. Use FlippedOmniglotWrapper.get_all instead.", DeprecationWarning)

    def flip(x):
        return 1 - x

    image_shape = (28, 28, 1)

    num_classes = 10

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            flip,
            transforms.Resize((image_shape[0], image_shape[1])),
        ]
    )

    train_dataset = datasets.Omniglot(
        DATAROOT, background=True, download=True, transform=transform
    )

    test_dataset = datasets.Omniglot(
        DATAROOT, background=False, download=True, transform=transform
    )

    return image_shape, num_classes, train_dataset, test_dataset


n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2**n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x


def one_hot_encode(target):
    """
    One hot encode with fixed 10 classes
    Args: target           - the target labels to one-hot encode
    Retn: one_hot_encoding - the OHE of this tensor
    """
    num_classes = 10
    one_hot_encoding = F.one_hot(torch.tensor(target), num_classes)

    return one_hot_encoding


class CIFAR10_Wrapper(DatasetWrapper):
    name = "cifar10"
    image_shape = (32, 32, 3)
    num_classes = 10
    pixel_range = (0.5, 0.5)

    @staticmethod
    def root(dataroot):
        return path.join(dataroot, "CIFAR10")

    @staticmethod
    def get_train(dataroot=DATAROOT, augment=True):
        if augment:
            transformations = [
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transformations = []

        transformations.extend([transforms.ToTensor(), preprocess])
        train_transform = transforms.Compose(transformations)

        train_dataset = datasets.CIFAR10(
            CIFAR10_Wrapper.root(dataroot),
            train=True,
            transform=train_transform,
            target_transform=one_hot_encode,
            download=True,
        )

        return train_dataset

    @staticmethod
    def get_test(dataroot=DATAROOT):

        test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

        return datasets.CIFAR10(
            CIFAR10_Wrapper.root(dataroot),
            train=False,
            transform=test_transform,
            target_transform=one_hot_encode,
            download=True,
        )


def get_CIFAR10(augment, dataroot, download):
    warnings.warn("DEPRECATED. Use CIFAR10_Wrapper.get_all instead.", DeprecationWarning)

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


class SVHN_Wrapper(DatasetWrapper):
    name = "svhn"
    image_shape = (32, 32, 3)
    num_classes = 10

    @staticmethod
    def root(dataroot):
        return path.join(dataroot, "SVHN")

    @staticmethod
    def get_train(dataroot=DATAROOT, augment=True):
        if augment:
            transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
        else:
            transformations = []

        transformations.extend([transforms.ToTensor(), preprocess])
        train_transform = transforms.Compose(transformations)

        return datasets.SVHN(
            SVHN_Wrapper.root(dataroot),
            split="train",
            transform=train_transform,
            target_transform=one_hot_encode,
            download=True,
        )

    @staticmethod
    def get_test(dataroot=DATAROOT):

        test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

        return datasets.SVHN(
            SVHN_Wrapper.root(dataroot),
            split="test",
            transform=test_transform,
            target_transform=one_hot_encode,
            download=True,
        )


def get_SVHN(augment, dataroot, download):
    warnings.warn("DEPRECATED. Use SVHN_Wrapper.get_all instead.", DeprecationWarning)

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

    print(f"dataroot: {dataroot}")
    print(f"path: {path}")

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


class Imagenet32_Wrapper(DatasetWrapper):
    name = "imagenet32"
    image_shape = (32, 32, 3)
    num_classes = NotImplementedError("currently labels for imagenet32 are unimplemented")

    @staticmethod
    def get_train(dataroot=DATAROOT):
        X_train_list = []
        for i in range(1, 11):
            X_train_batch = load_databatch(
                path.join(
                    dataroot,
                    "imagenet32_regular",
                    "train_32x32",
                    "train_data_batch_" + str(i),
                )
            )
            X_train_list.append(X_train_batch)

        X_train = np.concatenate(X_train_list)
        dummy_train_labels = torch.zeros(len(X_train))

        return torch.utils.data.TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32), dummy_train_labels
        )

    @staticmethod
    def get_test(dataroot=DATAROOT):
        X_test = load_databatch(
            path.join(
                dataroot,
                "imagenet32_regular",
                "valid_32x32",
                "val_data")
        )
        dummy_test_labels = torch.zeros(len(X_test))
        torch.utils.data.TensorDataset(
            torch.as_tensor(X_test, dtype=torch.float32), dummy_test_labels
        )


def get_imagenet32(dataroot):
    warnings.warn("DEPRECATED. Use Imagenet32_Wrapper.get_all instead.", DeprecationWarning)

    image_shape = (32, 32, 3)
    num_classes = None  # TODO @Sam

    X_train_list = []
    for i in range(1, 11):
        X_train_batch = load_databatch(
            path.join(
                dataroot,
                "data",
                "imagenet32_regular",
                "train_32x32",
                "train_data_batch_" + str(i),
            )
        )
        X_train_list.append(X_train_batch)

    X_train = np.concatenate(X_train_list)
    X_test = load_databatch(
        path.join(dataroot, "data", "imagenet32_regular", "valid_32x32", "val_data")
    )

    # TODO @Sam: you may want to replace this by another Dataset class.
    #  Note that TensorDataset will return a list of tensors for every batch, which contains exactly one element.
    dummy_train_labels = torch.zeros(len(X_train))
    dummy_test_labels = torch.zeros(len(X_test))

    train_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32), dummy_train_labels
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(X_test, dtype=torch.float32), dummy_test_labels
    )

    return image_shape, num_classes, train_dataset, test_dataset


def get_celeba(dataroot):
    warnings.warn("DEPRECATED. Use Imagenet32_Wrapper.get_all instead.", DeprecationWarning)

    class CelebA_LMBD_Wrapper:
        def __init__(self, lmdb_dataset):
            self.lmdb_dataset = lmdb_dataset

        def __len__(self):
            return self.lmdb_dataset.__len__()

        def __getitem__(self, item):
            sample = self.lmdb_dataset.__getitem__(item)
            sample = sample - 0.5
            return sample, torch.zeros(1)

    image_shape = (32, 32, 3)
    resize = 32

    num_classes = None  # TODO @Sam

    train_transform, valid_transform = _data_transforms_celeba64(resize)
    train_data = LMDBDataset(
        root=path.join(dataroot, "data/celeba64_lmdb"),
        name="celeba64",
        split="train",
        transform=train_transform,
        is_encoded=True,
    )
    valid_data = LMDBDataset(
        root=path.join(dataroot,"data/celeba64_lmdb"),
        name="celeba64",
        split="validation",
        transform=valid_transform,
        is_encoded=True,
    )
    test_data = LMDBDataset(
        root=path.join(dataroot, "data/celeba64_lmdb"),
        name="celeba64",
        split="test",
        transform=valid_transform,
        is_encoded=True,
    )

    return (
        image_shape,
        num_classes,
        CelebA_LMBD_Wrapper(train_data),
        CelebA_LMBD_Wrapper(test_data),
    )


# --------------


def flatten(outer):
    return [el for inner in outer for el in inner]


def load_databatch(path, img_size=32):
    """
    As copied from https://patrykchrabaszcz.github.io/Imagenet32/
    """
    d = unpickle_imagenet32(path)
    x = d["data"]  # is already uint8

    img_size2 = img_size * img_size
    x = np.dstack(
        (x[:, :img_size2], x[:, img_size2 : 2 * img_size2], x[:, 2 * img_size2 :])
    )
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(
        0, 3, 1, 2
    )  # (do not transpose, since we want (samples, 32, 32, 3))

    x = x / 256 - 0.5

    return x


def unpickle_imagenet32(file):
    """
    As copied from https://patrykchrabaszcz.github.io/Imagenet32/
    """
    with open(file, "rb") as fo:
        dict = pickle.load(fo)
    return dict


class CropCelebA64(object):
    """This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """

    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + "()"


def _data_transforms_celeba64(size):
    train_transform = transforms.Compose(
        [
            CropCelebA64(),
            transforms.Resize(size),
            # transforms.RandomHorizontalFlip(),  # taken out compared to NVAE --> we don't want data augmentation
            transforms.ToTensor(),
        ]
    )

    valid_transform = transforms.Compose(
        [
            CropCelebA64(),
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )

    return train_transform, valid_transform


class CelebaA_Wrapper(DatasetWrapper):
    name = "celeba"
    image_shape = (32, 32, 3)
    num_classes = NotImplementedError("currently labels for CelebaA are unimplemented")

    resize = 32
    train_transform, valid_transform = _data_transforms_celeba64(resize)

    @staticmethod
    def root(dataroot):
        return path.join(dataroot, "celeba64_lmdb")

    class CelebA_LMBD_Wrapper:
        def __init__(self, lmdb_dataset):
            self.lmdb_dataset = lmdb_dataset

        def __len__(self):
            return self.lmdb_dataset.__len__()

        def __getitem__(self, item):
            sample = self.lmdb_dataset.__getitem__(item)
            sample = sample - 0.5
            return sample, torch.zeros(1)

    @staticmethod
    def get_train(dataroot=DATAROOT):
        return LMDBDataset(
            root=CelebaA_Wrapper.root(dataroot),
            name="celeba64",
            split="train",
            transform=CelebaA_Wrapper.train_transform,
            is_encoded=True,
        )

    @staticmethod
    def get_test(dataroot=DATAROOT):
        return LMDBDataset(
            root=CelebaA_Wrapper.root(dataroot),
            name="celeba64",
            split="test",
            transform=CelebaA_Wrapper.train_transform,
            is_encoded=True,
        )


def num_samples(dataset, split):
    if dataset == "celeba":
        # return 27000 if train else 3000
        pass
    elif dataset == "celeba64":
        if split == "train":
            return 162770
        elif split == "validation":
            return 19867
        elif split == "test":
            return 19962
    else:
        raise NotImplementedError("dataset %s is unknown" % dataset)


class LMDBDataset(data.Dataset):
    def __init__(self, root, name="", split="train", transform=None, is_encoded=False):
        self.name = name
        self.split = split
        self.transform = transform
        if self.split in ["train", "validation", "test"]:
            lmdb_path = path.join(root, f"{self.split}.lmdb")
        else:
            print("invalid split param")
        self.data_lmdb = lmdb.open(
            lmdb_path,
            readonly=True,
            max_readers=1,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert("RGB")
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return num_samples(self.name, self.split)
