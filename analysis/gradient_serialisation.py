import torch
from torch.utils.data import DataLoader

from analysis.analysis_utils import load_glow_model, device, get_vanilla_test_dataset

GRADIENTS_DIR = "serialised_gradients/"


# def grad_dot_prod(delta_x, delta_y):
#     """Returns the dot product of two gradient vectors."""
#     return sum(
#         (grad_x * grad_y).sum() for grad_x, grad_y in zip(delta_x, delta_y)
#     )


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


def backprop_nll(batch):
    _, nll, _ = model(batch)
    model.zero_grad()
    nll.sum().backward()


def serialise_gradients(dataset, save_file, GradientStoreClass):
    print(f"creating {GRADIENTS_DIR + save_file}:")
    # grad_dict = {
    #     name: [] for name, _ in model.named_parameters()
    # }
    grad_store = GradientStoreClass(model)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print_update_every = len(dataset) // (20 * BATCH_SIZE)

    for i, batch in enumerate(dataloader):

        x, _ = batch

        x = x.to(device)

        backprop_nll(x)

        grad_store.extract_gradient_stats(model)
        # for name, p in model.named_parameters():
        #     grad_dict[name].append(
        #         (p.grad ** 2).sum()
        #     )

        if i % print_update_every == 0:
            print(f"{i * BATCH_SIZE}/{len(dataset)} complete")

    grad_store.serialise_gradient_stats(GRADIENTS_DIR + save_file)
    # for key, value in grad_dict.items():
    #     grad_dict[key] = torch.tensor(value)
    #
    # torch.save(grad_dict, GRADIENTS_DIR + save_file)
    print("done")


def get_save_file_name(model_name, dataset_name, batch_size):
    return f"trained_{model_name}_norms_{dataset_name}_{batch_size}.pt"


if __name__ == "__main__":

    MODEL_NAME = "celeba"

    MODEL_DIR = f"../glow_model/{MODEL_NAME}/"
    MODEL_FILE = "glow_checkpoint_419595.pt"

    model, hparams = load_glow_model(MODEL_DIR, MODEL_FILE)

    for BATCH_SIZE in [5]:
        for dataset_name in ["celeba", "svhn", "cifar", "imagenet32"]:
            dataset = get_vanilla_test_dataset(dataset_name)
            save_file = get_save_file_name(MODEL_NAME, dataset_name, BATCH_SIZE)
            serialise_gradients(dataset, save_file, L2NormStore)

