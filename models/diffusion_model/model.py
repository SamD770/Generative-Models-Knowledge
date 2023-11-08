from generative_model import GenerativeModel

from path_definitions import DIFFUSION_ROOT
import re

from data.datasets import get_celeba
from data.utils import get_dataset
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from os import path
from torch import nn

import torch

"""
USING THE CODE FROM: https://github.com/lucidrains/denoising-diffusion-pytorch
"""

DEFAULT_TIMESTEPS = 1000
DEFAULT_SAMPLES_EVAL = 1

# The pattern that should match the name is model_name_T_timesteps_n_samples
#   These values are only used at evaluation time, so the same model weights are loaded.

timesteps_regex = re.compile("(.*)_(\d*)_timesteps")
samples_regex = re.compile("(.*)_(\d*)_samples")

# TODO: re-defining is a quick fix. need to put in one module and import everywhere
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class DiffusionModel(GenerativeModel, nn.Module):
    """."""
    def __init__(self, image_shape=(32, 32, 3), timesteps_eval=DEFAULT_TIMESTEPS, samples_eval=DEFAULT_SAMPLES_EVAL):

        super().__init__()

        input_width, input_height, input_channels = image_shape

        self.unet = Unet(
            dim=input_width,
            dim_mults=(1, 2, 4, 8)
        )

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=input_width,
            timesteps=DEFAULT_TIMESTEPS,  # number of steps
            loss_type='l1'  # L1 or L2
        )

        # timesteps_eval is the number of diffusion steps used to evaluate the variational lower bound in eval_nll
        # samples eval is the number of independent noise samples used to approximate the VLB
        self.timesteps_eval = timesteps_eval
        self.samples_eval = samples_eval

    def eval_nll(self, x):
        x += 0.5                    # diffusion model implementation works on images (0, 1)

        # Computes an empirical estimate of the Variational Lower Bound
        vlb_estimates = []

        for _ in range(self.samples_eval):

            if self.timesteps_eval == DEFAULT_TIMESTEPS: # This uniformly samples t from [0, 1 ... t-1]
                vlb_sample = self.diffusion(x)

            else:
                b, c, h, w = x.shape

                # passing the value of t runs the diffusion process to x_{t+1}, for this reason we have to subtract 1
                t = torch.ones((b,), device=device).long()*(self.timesteps_eval - 1)
                x = self.diffusion.normalize(x)
                vlb_sample = self.diffusion.p_losses(x, t)

            vlb_estimates.append(vlb_sample)

        return sum(vlb_estimates) / len(vlb_estimates)

    def generate_sample(self, batch_size):
        imgs = self.diffusion.sample(batch_size=batch_size)
        return imgs - 0.5           # diffusion model implementation works on images (0, 1)

    @staticmethod
    def load_serialised(model_name):

        # greps the number of evaluation timesteps & samples out of the model_name,
        # for example to load my_diffusion_model.pt such that it evaluates using exactly 6 timesteps and 7 samples, pass:
        # model_name = my_diffusion_model_6_timesteps_7_samples

        # ordering is important (this is to ensure compute isn't wasted by re-computing summary statistics)

        search_result = samples_regex.match(model_name)
        if search_result:
            samples_eval = int(search_result.group(2))
            model_name = search_result.group(1)
        else:
            samples_eval = DEFAULT_SAMPLES_EVAL

        model_name, timesteps_eval = extract_timesteps(model_name)

        save_path = get_save_path(model_name)
        checkpoint = torch.load(save_path)

        print(f"Loading {model_name} trained for {checkpoint['epoch']} epochs")

        diffusion_model = DiffusionModel(
            timesteps_eval=timesteps_eval,
            samples_eval=samples_eval
        )

        diffusion_model.diffusion.load_state_dict(
            checkpoint["diffusion_state_dict"]
        )

        return diffusion_model


def extract_timesteps(model_name):
    search_result = timesteps_regex.match(model_name)
    if search_result:
        timesteps_eval = int(search_result.group(2))
        model_name = search_result.group(1)
    else:
        timesteps_eval = DEFAULT_TIMESTEPS
    return model_name, timesteps_eval


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



def test_qsample():
    model = DiffusionModel.load_serialised(f"diffusion_cifar10_1_timesteps")
    print(f"{model.diffusion.objective=}")

    model.diffusion.to(device)

    # print(model.diffusion.sqrt_alphas_cumprod[:30])

    t = torch.arange(30, device=device)*30
    x_start = torch.ones(30, device=device).resize(30, 1, 1, 1)
    print(model.diffusion.q_sample(x_start, t).resize(30))

    model.eval_nll(x_start)


if __name__ == "__main__":

    print(device)
    test_qsample()
    exit()

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

