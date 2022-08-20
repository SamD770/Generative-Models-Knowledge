from generative_model import GenerativeModel

import torch

from torch import nn
# from torchvision import datasets, transforms
from itertools import chain

from data.datasets import get_FashionMNIST

import matplotlib.pyplot as plt




class SimpleVAE(nn.Module, GenerativeModel):
    def __init__(self, input_shape=(32, 32, 3), latent_dims=10):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dims = latent_dims

        self.encoder = Encoder(input_shape, latent_dims)
        self.decoder = Decoder(input_shape, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # TODO: write this to be non-device specific
        self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        pass

    def get_normal_sample(self, means, stds):
        return means + stds * self.N.sample((self.latent_dims, ))

    def eval_nll(self, x):
        # Return the ELBO (a lower bound on the NLL)
        mu, sigma = self.encoder(x)

        kl_divergence = (mu ** 2 + sigma ** 2 - 1) / 2 - torch.log(sigma)
        kl_divergence = kl_divergence.sum()

        latents = self.get_normal_sample(mu, sigma)

        reconstructions = self.decoder(latents)

        reconstruction_loss = (x - reconstructions) ** 2
        reconstruction_loss = reconstruction_loss.sum()

        ELBO = reconstruction_loss + kl_divergence

        return ELBO

    def generate_sample(self, batch_size):

        latents = self.N.sample((batch_size, self.latent_dims))
        samples = self.decoder(latents)

        return samples

    @staticmethod
    def load_serialised(save_dir, save_file, **params):
        checkpoint = torch.load(save_dir + save_file)

        vae = SimpleVAE(**params)
        vae.load_state_dict(checkpoint["vae_state_dict"])

        return vae


class Encoder(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), latent_dims=10):
        super().__init__()

        input_width, input_height, input_channels = input_shape

        self.conv_1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=(2, 2))
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv_layers = [self.conv_1, self.conv_2, self.conv_3]
        self.intermediate_conv_size = 64 * input_width//2 * input_width//2

        self.linear_1 = nn.Linear(self.intermediate_conv_size, 32)
        self.mean_head = nn.Linear(32, latent_dims)
        self.std_head = nn.Linear(32, latent_dims)

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = torch.relu(x)

        x = x.view(-1, self.intermediate_conv_size)
        x = self.linear_1(x)
        x = torch.relu(x)
        mu = self.mean_head(x)
        sigma = torch.exp(self.std_head(x))
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, input_shape=(32, 32, 3), latent_dims=10):
        super().__init__()
        input_width, input_height, input_channels = input_shape

        self.inter_input_width = input_width//2
        self.inter_input_height = input_height//2

        self.linear_1 = nn.Linear(latent_dims, 32)
        self.linear_2 = nn.Linear(32, 32 * (self.inter_input_width) * (self.inter_input_height))
        self.upsample = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=(2, 2))
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = torch.relu(x)

        x = x.view(-1, 32, self.inter_input_width, self.inter_input_height)
        x = self.upsample(x)

        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x

#
# N = torch.distributions.Normal(0, 1)
# # N.loc = N.loc.cuda()  # hack to get sampling on the GPU
# # N.scale = N.scale.cuda()

#
# def get_sample(means, stds):
#     n = means.shape
#     return means + stds * N.sample(n)


def training_loop(n_epochs, optimizer, vae, train_loader, checkpoint_path, device, starting_epoch=1):
    for n in range(starting_epoch, n_epochs + 1):
        train_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device=device)

            loss = vae.eval_nll(imgs)

            # mu, sigma = encoder(imgs)
            #
            # kl_divergence = (mu ** 2 + sigma ** 2 - 1) / 2 - torch.log(sigma)
            # kl_divergence = kl_divergence.sum()
            #
            # latents = get_sample(mu, sigma)
            #
            # reconstructions = decoder(latents)

            # reconstruction_loss = (imgs - reconstructions) ** 2
            # reconstruction_loss = reconstruction_loss.sum()
            #
            # loss = reconstruction_loss + kl_divergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if n < 5 or n % 10 == 0:
            print(f"epoch: {n}, train loss: {train_loss / len(train_loader)}")
            torch.save({
                "epoch": n,
                "vae_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, checkpoint_path)



if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    input_shape, num_classes, train_dataset, test_dataset = get_FashionMNIST(dataroot="../data/")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

    print(input_shape)

    my_vae = SimpleVAE(input_shape=input_shape)

    my_optimizer = torch.optim.Adam(my_vae.parameters(), lr=3e-4)

    my_vae.to(device=device)

    training_loop(
        n_epochs=100,
        optimizer=my_optimizer,
        vae=my_vae,
        train_loader=train_loader,
        checkpoint_path="VAE_FashionMNIST_checkpoint.pt",
        device=device
    )

#
#
#
# my_encoder = Encoder().to(device=device)
# my_decoder = Decoder().to(device=device)
#
#
# print(my_encoder)
# print(my_decoder)

#



