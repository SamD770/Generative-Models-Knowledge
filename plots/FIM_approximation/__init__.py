import random
from datetime import datetime
from itertools import product

import torch
from torch.utils.data import DataLoader

from data.utils import SampleDataset
from models.utils import load_generative_model

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)


class MultiLayerFIMStore:
    def __init__(self, model, layer_names, weight_indices):
        self.model = model
        self.layer_names = layer_names
        self.weight_indices = weight_indices

        self.n = 0
        self.outer_product_windows = [[0. for _ in layer_names] for _ in layer_names]

    def get_grad_vecs(self, target_model):

        grad_vecs = []

        for layer_name, indices in zip(self.layer_names, self.weight_indices):

            for name, p in target_model.named_parameters():

                if name == layer_name:
                    grad_vec = torch.flatten(p.grad)

                    grad_vec = grad_vec.index_select(dim=0, index=indices)
                    grad_vec = grad_vec.nan_to_num()

                    grad_vecs.append(grad_vec)
                    break

        # exit()
        return grad_vecs

    def extract_gradient_stats(self, target_model):
        grad_vecs = self.get_grad_vecs(target_model)

        for (i, v_i), (j, v_j) in product(enumerate(grad_vecs), enumerate(grad_vecs)):

            self.outer_product_windows[i][j] += torch.outer(v_i, v_j)

        self.n += 1

    def get_windows(self):
        """Divides each row by n to get the actual empirical FIM."""

        FIM_windows = []
        for row in self.outer_product_windows:
            new_row = []
            for window in row:
                new_row.append(window/self.n)

            FIM_windows.append(new_row)

        return FIM_windows

    @staticmethod
    def from_model(model, n_layers, n_weights_per_layer):

        params = list(model.named_parameters())

        selected_layers = random.sample(params, k=n_layers)

        # Sort such that layers with smaller parameter values are at the top of the plot
        selected_layers = sorted(selected_layers, key=(lambda x: x[1].mean()))

        selected_layer_names = [
            name for name, _ in selected_layers
        ]

        selected_indices_list = []
        n_selected_weights_list = []

        for layer_name, p in selected_layers:

            n_weights = p.numel()

            all_indices = range(n_weights)
            n_selected_weights = min(n_weights_per_layer, n_weights)
            n_selected_weights_list.append(n_selected_weights)

            selected_weight_indices = sorted(random.sample(all_indices, k=n_selected_weights))

            selected_weight_indices = torch.tensor(selected_weight_indices).to(device)
            selected_indices_list.append(selected_weight_indices)

            # param_vec = torch.flatten(p)
            #
            # print(param_vec.index_select(dim=0, index=selected_weight_indices))

            print(layer_name, p.shape)

        fim_store = MultiLayerFIMStore(model, selected_layer_names, selected_indices_list)
        return fim_store, n_selected_weights_list


def backprop_nll(model, batch):
    nll = model.eval_nll(batch)
    model.zero_grad()
    nll.sum().backward()


device = torch.device("cuda")


def compute_FIM(fim_store, model, sample_dataset):
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        sample_dataset, batch_size=1, shuffle=False, drop_last=True
    )

    print_update_every = len(sample_dataset) // (20)

    for i, batch in enumerate(dataloader):
        x, y = batch

        x = x.to(device)

        backprop_nll(model, x)

        fim_store.extract_gradient_stats(model)

        if i % print_update_every == 0:
            print(f"({datetime.now()}) {i}/{len(sample_dataset)} complete")

    return fim_store.get_windows()


def load_prerequisites(model_name, model_type, n_layers, n_weights_per_layer, sampling_model_name, batch_count=32):
    # Load models for computing the FIM
    model = load_generative_model(model_type, model_name)
    model.to(device)

    if sampling_model_name is None:
        sampling_model = model
    else:
        sampling_model = load_generative_model(model_type, model_name)
        sampling_model.to(device)

    sample_dataset = SampleDataset(sampling_model, batch_count=batch_count)
    fim_store, n_selected_weights_list = MultiLayerFIMStore.from_model(model, n_layers, n_weights_per_layer)

    return fim_store, model, n_selected_weights_list, sample_dataset
