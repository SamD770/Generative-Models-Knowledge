import download_model
from datasets import get_SVHN, get_CIFAR10
import torch


device = "cuda"

svhn = get_SVHN(augment=False, dataroot="../", download=False)
input_size, num_classes, train_dataset, test_dataset = svhn

my_VAE = VAE_model.PlainVAE(
    in_dim,
    z_dim,
    encode_layer_dims,
    decode_layer_dims,
    likelihoods,
    dataset,
    init_lr
)


def training_loop(model, dataloader, optimizer, n_epochs):

    for epoch_no in range(1, n_epochs+1):
        loss_train = 0
        for imgs, labels in dataloader:
            imgs = imgs.to(device)

            params_p_x_z, q_z_x, z_sample_q_z_x = model(imgs)
            loss, _ = model.compute_loss(params_p_x_z, q_z_x, z_sample_q_z_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
