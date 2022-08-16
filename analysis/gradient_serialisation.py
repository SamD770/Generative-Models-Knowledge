import torch
from torch.utils.data import DataLoader, Subset
from datetime import datetime

import random

from analysis.analysis_utils import load_glow_model, device, get_vanilla_test_dataset, SampleDataset

GRADIENTS_DIR = "serialised_gradients/"


class GradientStore:
    def __init__(self, target_model):
        self.grad_dict = self.setup_grad_dict(target_model)

    def setup_grad_dict(self, target_model):
        raise NotImplementedError()

    def extract_gradient_stats(self, target_model):
        raise NotImplementedError()

    def serialise_gradient_stats(self, save_file):
        raise NotImplementedError()


class L2NormStore(GradientStore):
    """Stores a mapping from the layer name to L^2 norm of the gradient vector."""
    def setup_grad_dict(self, target_model):
        return {
            name: [] for name, _ in target_model.named_parameters()
        }

    def extract_gradient_stats(self, target_model):
        for name, p in target_model.named_parameters():
            self.grad_dict[name].append(
                (p.grad ** 2).sum()
            )

    def serialise_gradient_stats(self, save_file_dir):
        for key, value in self.grad_dict.items():
            self.grad_dict[key] = torch.tensor(value)

        torch.save(self.grad_dict, save_file_dir)


class SingleGradStore(GradientStore):
    """Stores the value for one subset of parameters"""
    def __init__(self, target_model, param_name, index):
        self.param_name = param_name
        self.index = index
        super().__init__(target_model)

    def setup_grad_dict(self, target_model):
        return []

    def extract_gradient_stats(self, target_model):
        for name, p in target_model.named_parameters():
            if name == self.param_name:
                param_grad = p.grad[self.index].clone()
                self.grad_dict.append(param_grad)
                # print("L^2: ", (param_grad**2).sum().item(), "param:", (p**2).sum().item())

    def serialise_gradient_stats(self, save_file_dir):
        grad_t = torch.tensor(self.grad_dict)
        torch.save(grad_t, save_file_dir)
        self.grad_dict = []


def backprop_nll(batch):
    _, nll, _ = model(batch)
    model.zero_grad()
    nll.sum().backward()


def serialise_gradients(dataset, save_file, grad_store):
    print(f"creating {GRADIENTS_DIR + save_file}:")
    # grad_dict = {
    #     name: [] for name, _ in model.named_parameters()
    # }

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print_update_every = len(dataset) // (20 * BATCH_SIZE)

    for i, batch in enumerate(dataloader):

        x, y = batch

        x = x.to(device)

        backprop_nll(x)

        grad_store.extract_gradient_stats(model)

        if i % print_update_every == 0:
            print(f"({datetime.now()}) {i * BATCH_SIZE}/{len(dataset)} complete")

    grad_store.serialise_gradient_stats(GRADIENTS_DIR + save_file)

    print("done")


def serialise_FIM(save_file, layer=None):
    # layer = "flow.layers.100.actnorm.bias"

    class FIMStore(GradientStore):
        """Keeps a stored representation of the FIM of a given model"""
        def __init__(self, target_model):
            super().__init__(target_model)
            self.old_FIM = None
            self.n = 0

        def setup_grad_dict(self, target_model):
            if layer is None:
                return {
                    name: 0 for name, _ in target_model.named_parameters()
                }
            else:
                return {
                    layer: 0
                }

        def extract_gradient_stats(self, target_model):
            for name, p in target_model.named_parameters():
                if name == layer or layer is None:

                    grad_vec = torch.flatten(p.grad)
                    fim_approximation = torch.outer(grad_vec, grad_vec)

                    self.grad_dict[name] += fim_approximation
            self.n += 1

                    # new_FIM = self.grad_dict[layer] / self.n
                    #
                    # if self.old_FIM is not None:
                    #     delta = torch.abs(self.old_FIM - new_FIM)
                    #     if self.n % 100 == 0:
                    #         print(f"n: {self.n} delta: {torch.mean(delta)}")
                    #
                    # self.old_FIM = new_FIM

        def serialise_gradient_stats(self, save_file_dir):
            for model_layer, fim_sum in self.grad_dict.items():
                self.grad_dict[model_layer] = fim_sum/self.n

            # print("FIM:", self.grad_dict[layer][:10, :10])
            # print("average size:", self.old_FIM.abs().mean())
            #
            # torch.save(self.grad_dict, save_file_dir)

    sample_dataset = SampleDataset(model, batch_count=512)
    serialise_gradients(sample_dataset, save_file, FIMStore)


def get_save_file_name(model_name, dataset_name, batch_size, method="norms"):
    return f"trained_{model_name}_{method}_{dataset_name}_{batch_size}.pt"


if __name__ == "__main__":

    MODEL_NAME = "cifar_long"

    MODEL_DIR = f"../glow_model/{MODEL_NAME}/"
    MODEL_FILE = "glow_checkpoint_585750.pt"

    model, hparams = load_glow_model(MODEL_DIR, MODEL_FILE)

    NUM_SAMPLES = 1000


    param_list = [
        (name, p.shape, len(p.flatten())) for name, p in model.named_parameters()
    ]

    # select a random layer
    param_size = 0
    while param_size < 100:
        my_param_name, my_param_shape, param_size = random.choice(param_list)

    # select a random parameter from this layer
    my_index = tuple(
        random.randrange(i) for i in my_param_shape
    )

    my_param_name = "flow.layers.98.invconv.lower"
    my_index = (31, 7)

    print("chosen:", my_param_name, my_index)

    for BATCH_SIZE in [1]:
        for dataset_name in ["cifar", "svhn", "celeba", "imagenet32"]:
            dataset = get_vanilla_test_dataset(dataset_name)

            if NUM_SAMPLES is not None:
                dataset = Subset(dataset, range(NUM_SAMPLES))

            save_file = get_save_file_name(MODEL_NAME, dataset_name, BATCH_SIZE, method="single_grad")

            grad_store = SingleGradStore(model, my_param_name, my_index)

            serialise_gradients(dataset, save_file, grad_store)

