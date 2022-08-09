import torch
from torch.utils.data import DataLoader

from analysis.analysis_utils import load_glow_model, device, get_vanilla_test_dataset


MODEL_NAME = "cifar_long"

MODEL_DIR = f"../glow_model/{MODEL_NAME}/"
MODEL_FILE = "glow_checkpoint_585750.pt"

GRADIENTS_DIR = "serialised_gradients/"


BATCH_SIZE = 32


def backprop_nll(batch):
    _, nll, _ = model(batch)
    model.zero_grad()
    nll.sum().backward()


def grad_dot_prod(delta_x, delta_y):
    """Returns the dot product of two gradient vectors."""
    return sum(
        (grad_x * grad_y).sum() for grad_x, grad_y in zip(delta_x, delta_y)
    )


def serialise_gradients(dataset, save_file):
    print(f"creating {GRADIENTS_DIR + save_file}:")
    grad_dict = {
        name: [] for name, _ in model.named_parameters()
    }

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print_update_every = len(dataset) // (20 * 32)

    for i, batch in enumerate(dataloader):

        x, _ = batch

        x = x.to(device)

        backprop_nll(x)

        for name, p in model.named_parameters():
            grad_dict[name].append(
                (p.grad ** 2).sum()
            )

        if i % print_update_every == 0:
            print(f"{i * BATCH_SIZE}/{len(dataset)} complete")

    for key, value in grad_dict.items():
        grad_dict[key] = torch.tensor(value)

    torch.save(grad_dict, GRADIENTS_DIR + save_file)
    print("done")


def get_save_file_name(model_name, dataset_name, batch_size):
    return f"trained_{model_name}_norms_{dataset_name}_{batch_size}.pt"


if __name__ == "__main__":
    model, hparams = load_glow_model(MODEL_DIR, MODEL_FILE)

    for dataset_name in ["cifar", "svhn", "celeba", "imagenet32"]:
        dataset = get_vanilla_test_dataset(dataset_name)
        save_file = get_save_file_name(MODEL_NAME, dataset_name, BATCH_SIZE)
        serialise_gradients(dataset, save_file)

