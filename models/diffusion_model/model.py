from generative_model import GenerativeModel

from path_definitions import DIFFUSION_ROOT

from data.datasets import get_celeba
from data.utils import get_dataset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from os import path
from torch import nn

import torch

"""
USING THE CODE FROM: https://github.com/lucidrains/denoising-diffusion-pytorch
"""


class DiffusionModel(GenerativeModel, nn.Module):
    """."""
    def __init__(self, image_shape=(32, 32, 3)):

        super().__init__()

        input_width, input_height, input_channels = image_shape

        self.unet = Unet(
            dim=input_width,
            dim_mults=(1, 2, 4, 8)
        )

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=input_width,
            timesteps=1000,  # number of steps
            loss_type='l1'  # L1 or L2
        )

    def eval_nll(self, x):
        x += 0.5                    # diffusion model implementation works on images (0, 1)
        return self.diffusion(x)

    def generate_sample(self, batch_size):
        imgs = self.diffusion.sample(batch_size=batch_size)
        return imgs - 0.5           # diffusion model implementation works on images (0, 1)

    @staticmethod
    def load_serialised(model_name):
        save_path = get_save_path(model_name)
        checkpoint = torch.load(save_path)

        print(f"Loading {model_name} trained for {checkpoint['epoch']} epochs")

        diffusion_model = DiffusionModel()
        diffusion_model.diffusion.load_state_dict(
            checkpoint["diffusion_state_dict"])

        return diffusion_model


def get_save_path(model_name):
    return path.join(DIFFUSION_ROOT, model_name + ".pt")


def training_loop(
    n_epochs, optimizer, model, train_loader, checkpoint_path, device, starting_epoch=1
):
    for n in range(starting_epoch, starting_epoch + n_epochs + 1):
        train_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device=device)

            loss = model.eval_nll(imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if n < 10:
            print(f"epoch: {n}, train loss: {train_loss / len(train_loader)}")
            torch.save(
                {
                    "epoch": n,
                    "diffusion_state_dict": model.diffusion.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(device)

    for dataset_name in ["gtsrb", "cifar10", "svhn", "imagenet32"]:
        train_dataset = get_dataset(dataset_name, split="train")

        # input_shape, _, train_dataset, _ = get_celeba(dataroot="./")
        # dataset_name = "celeba"


        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True
        )

        model = DiffusionModel.load_serialised(f"diffusion_{dataset_name}")

        model.diffusion.to(device)

        my_optimizer = torch.optim.Adam(model.diffusion.parameters(), lr=3e-4)

        print("training on ", dataset_name)

        training_loop(
            n_epochs=5,
            optimizer=my_optimizer,
            model=model,
            train_loader=train_loader,
            checkpoint_path=get_save_path(f"diffusion_{dataset_name}"),
            device=device,
            starting_epoch=5
        )

