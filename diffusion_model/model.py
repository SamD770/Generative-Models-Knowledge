from generative_model import GenerativeModel

from path_definitions import DIFFUSION_ROOT

from data.datasets import get_celeba
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

import torch

"""
USING THE CODE FROM: https://github.com/lucidrains/denoising-diffusion-pytorch
"""


class DiffusionModel(GenerativeModel):
    """."""
    def __init__(self, image_shape=(32, 32, 3)):

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
    def load_serialised(save_file, save_dir=DIFFUSION_ROOT, **params):
        raise NotImplementedError()



def training_loop(
    n_epochs, optimizer, model, train_loader, checkpoint_path, device, starting_epoch=1
):
    for n in range(starting_epoch, n_epochs + 1):
        train_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device=device)

            loss = model.eval_nll(imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if n < 5 or n % 10 == 0:
            print(f"epoch: {n}, train loss: {train_loss / len(train_loader)}")
            torch.save(
                {
                    "epoch": n,
                    "vae_state_dict": model.diffusion.state_dict(),
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

    input_shape, _, train_dataset, _ = get_celeba(dataroot="./")
    dataset_name = "celeba"

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True
    )

    model = DiffusionModel()

    model.diffusion.to(device)

    my_optimizer = torch.optim.Adam(model.diffusion.parameters(), lr=3e-4)

    training_loop(
        n_epochs=100,
        optimizer=my_optimizer,
        model=model,
        train_loader=train_loader,
        checkpoint_path=f"./diffusion_model/VAE_{dataset_name}.pt",
        device=device,
    )

