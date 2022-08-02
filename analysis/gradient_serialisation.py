import torch
from torch.utils.data import DataLoader

from analysis.analysis_utils import load_glow_model, vanilla_test_cifar, vanilla_test_svhn, device


MODEL_DIR = "../glow_model/cifar_glow/"
MODEL_FILE = "glow_checkpoint_195250.pt"

GRADIENTS_DIR = "serialised_gradients/"


model, hparams = load_glow_model(MODEL_DIR, MODEL_FILE)
batch_size = 32


def backprop_nll(batch):
    _, nll, _ = model(batch)
    model.zero_grad()
    nll.sum().backward()


def grad_dot_prod(delta_x, delta_y):
    """Returns the dot product of two gradient vectors."""
    return sum(
        (grad_x * grad_y).sum() for grad_x, grad_y in zip(delta_x, delta_y)
    )


# TODO: add likelihoods and full gradient vectors

for dataset, save_file in zip([vanilla_test_cifar(), vanilla_test_svhn()],
                              [f"cifar_id_norms_{batch_size}.pt", f"cifar_od_norms_{batch_size}.pt"]):

    print(f"creating {GRADIENTS_DIR + save_file}:")
    grad_dict = {
        name: [] for name, _ in model.named_parameters()
    }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
            print(f"{i*batch_size}/{len(dataset)} complete")

    for key, value in grad_dict.items():
        grad_dict[key] = torch.tensor(value)

    torch.save(grad_dict, GRADIENTS_DIR + save_file)
    print("done")


"""    
for i in range(gradient_count):
        img, _ = dataset[i]
        backprop_nll(img)

        for name, p in model.named_parameters():
            grad_dict[name].append(
                (p.grad**2).sum()
            )

        if i % (gradient_count//50) == 0:
            print(f"{i}/{gradient_count} complete")

    for key, value in grad_dict.items():
        grad_dict[key] = torch.tensor(value)

    torch.save(grad_dict, save_file)
    print("done")
"""
