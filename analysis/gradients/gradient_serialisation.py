import torch
from torch.utils.data import DataLoader, Subset
from datetime import datetime

import matplotlib.pyplot as plt

import random

from analysis.analysis_utils import load_generative_model, device, get_vanilla_test_dataset, SampleDataset
from torch.nn.functional import normalize

GRADIENTS_DIR = "./analysis/gradients/serialised_gradients/"


class GradientStore:
    def __init__(self, target_model):
        self.grad_store = self.setup_grad_store(target_model)

    def setup_grad_store(self, target_model):
        raise NotImplementedError()

    def extract_gradient_stats(self, target_model):
        raise NotImplementedError()

    def serialise_gradient_stats(self, save_file):
        raise NotImplementedError()


class NormStore(GradientStore):
    """Stores a mapping from the layer name to norm of the gradient vector for some norm."""
    def setup_grad_store(self, target_model):
        return {
            name: [] for name, _ in target_model.named_parameters()
        }

    def extract_gradient_stats(self, target_model):
        for name, p in target_model.named_parameters():
            self.grad_store[name].append(
                self.take_norm(p.grad, name)
            )

    def serialise_gradient_stats(self, save_file_dir):
        for key, value in self.grad_store.items():
            self.grad_store[key] = torch.tensor(value)

        torch.save(self.grad_store, save_file_dir)

    def take_norm(self, gradient_vector, layer):
        raise NotImplementedError()


class L2NormStore(NormStore):
    """Stores a mapping from the layer name to L^2 norm of the gradient vector."""
    def take_norm(self, gradient_vector, layer):
        return (gradient_vector**2).sum()


class FisherNormStore(NormStore):
    """Stores a mapping from the layer name to u^T (F)^(-1) u where F is the Fisher Information Matrix and
     u is the gradient vector."""
    def __init__(self, target_model, FIM_inv):
        super().__init__(target_model)
        self.FIM_inv = FIM_inv

    def take_norm(self, gradient_vector, layer):
        gradient_vector * (self.FIM_inv) * gradient_vector # TODO: figure out which of these need to be


class SingleGradStore(GradientStore):
    """Stores the value for one subset of parameters"""
    def __init__(self, target_model, param_name, index):
        self.param_name = param_name
        self.index = index
        super().__init__(target_model)

    def setup_grad_store(self, target_model):
        return []

    def extract_gradient_stats(self, target_model):
        for name, p in target_model.named_parameters():
            if name == self.param_name:
                param_grad = p.grad[self.index].clone()
                self.grad_dict.append(param_grad)
                # print("L^2: ", (param_grad**2).sum().item(), "param:", (p**2).sum().item())

    def serialise_gradient_stats(self, save_file_dir):
        grad_t = torch.tensor(self.grad_store)
        torch.save(grad_t, save_file_dir)
        self.grad_dict = []


class FIMStore(GradientStore):
    """Keeps a stored representation of the FIM of a given model"""
    def __init__(self, target_model):
        super().__init__(target_model)
        self.old_FIM = None
        self.n = 0

    def setup_grad_store(self, target_model):
        return 0

    def extract_gradient_stats(self, target_model):
        grad_vec = self.get_grad_vec(target_model)
        fim_approximation = torch.outer(grad_vec, grad_vec)
        self.grad_store += fim_approximation
        self.n += 1

        new_FIM = self.grad_store / self.n

        if self.old_FIM is not None:
            delta = torch.abs(self.old_FIM - new_FIM)
            if self.n % 100 == 0:
                print(f"n: {self.n} delta: {torch.mean(delta)}")

        self.old_FIM = new_FIM

    def serialise_gradient_stats(self, save_file_dir):
        self.grad_store = self.grad_store / self.n

        print("FIM:", self.grad_store[:10, :10])
        print("average size:", self.old_FIM.abs().mean())

        normed_FIM = normalize(self.grad_store, dim=0, p=1)

        print("normed FIM:", normed_FIM)

        print()

        plt.imshow(normed_FIM.cpu())

        plt.title(save_file_dir)
        # plt.axes("off")
        plt.savefig(save_file_dir)



        # torch.save(self.grad_store, save_file_dir)

    def get_grad_vec(self, target_model):
        raise NotImplementedError()


class RandomFIMStore(FIMStore):
    pass


class LayerFIMStore(FIMStore):
    def __init__(self, target_model, layer_name):
        super().__init__(target_model)
        self.layer_name = layer_name

    def get_grad_vec(self, target_model):
        for name, p in target_model.named_parameters():
            if name == self.layer_name:
                grad_vec = torch.flatten(p.grad)
                return grad_vec


def backprop_nll(model, batch):
    nll = model.eval_nll(batch)
    model.zero_grad()
    nll.sum().backward()


def serialise_gradients(model, dataset, save_file, grad_store, batch_size, save_dir=GRADIENTS_DIR):
    print(f"creating {save_dir + save_file}:")
    # grad_dict = {
    #     name: [] for name, _ in model.named_parameters()
    # }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print_update_every = len(dataset) // (20 * batch_size)

    for i, batch in enumerate(dataloader):

        x, y = batch

        x = x.to(device)

        backprop_nll(model, x)

        grad_store.extract_gradient_stats(model)

        if i % print_update_every == 0:
            print(f"({datetime.now()}) {i * batch_size}/{len(dataset)} complete")

    grad_store.serialise_gradient_stats(save_dir + save_file)

    print("done")
#
#
# def serialise_FIM(save_file, layer=None):
#
#     sample_dataset = SampleDataset(model, batch_count=512)
#     serialise_gradients(sample_dataset, save_file, FIMStore)


def get_save_file_name(model_name, dataset_name, batch_size, method="norms", filetype="pt"):
    return f"trained_{model_name}_{method}_{dataset_name}_{batch_size}.{filetype}"


if __name__ == "__main__":

    # model = load_generative_model("glow", ".../glow_model/FashionMNIST_stable/",
    #                               "glow_checkpoint_18740.pt", image_shape=(28, 28, 1))

    # model = load_generative_model("PixelCNN", "../../pixelCNN_model/", "PixelCNN_new_checkpoint.pt") # PixelCNN_FashionMNIST_checkpoint.pt")

    for BATCH_SIZE in [2]:
        for dataset_name in ["FashionMNIST", "MNIST"]:
            dataset = get_vanilla_test_dataset(dataset_name, dataroot="../../")
            model_name = "PixelCNN_FashionMNIST"
            save_file_name = get_save_file_name(model_name, dataset_name, BATCH_SIZE)

            grad_store = L2NormStore(model)

            serialise_gradients(dataset, save_file_name, grad_store)

    # MODEL_NAME = "cifar_long"
    #
    # MODEL_DIR = f"../glow_model/{MODEL_NAME}/"
    # MODEL_FILE = "glow_checkpoint_585750.pt"

    # model, hparams = load_glow_model(MODEL_DIR, MODEL_FILE)
    #
    # NUM_SAMPLES = 1000
    #
    #
    # param_list = [
    #     (name, p.shape, len(p.flatten())) for name, p in model.named_parameters()
    # ]
    #
    # # select a random layer
    # param_size = 0
    # while param_size < 100:
    #     my_param_name, my_param_shape, param_size = random.choice(param_list)
    #
    # # select a random parameter from this layer
    # my_index = tuple(
    #     random.randrange(i) for i in my_param_shape
    # )
    #
    # my_param_name = "flow.layers.98.invconv.lower"
    # my_index = (31, 7)
    #
    # print("chosen:", my_param_name, my_index)
    #
    # for BATCH_SIZE in [1]:
    #     for dataset_name in ["cifar", "svhn", "celeba", "imagenet32"]:
    #         dataset = get_vanilla_test_dataset(dataset_name)
    #
    #         if NUM_SAMPLES is not None:
    #             dataset = Subset(dataset, range(NUM_SAMPLES))
    #
    #         save_file = get_save_file_name(MODEL_NAME, dataset_name, BATCH_SIZE, method="single_grad")
    #
    #         grad_store = SingleGradStore(model, my_param_name, my_index)
    #
    #         serialise_gradients(dataset, save_file, grad_store)

