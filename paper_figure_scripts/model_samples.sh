python -m plots.model_samples.array \
    glow svhn_working celeba gtsrb_glow_continued cifar_long imagenet32

python -m plots.model_samples.array \
    diffusion \
      diffusion_svhn \
      diffusion_celeba \
      diffusion_gtsrb \
      diffusion_cifar10 \
      diffusion_imagenet32

python -m plots.model_samples.array \
    vae VAE_svhn VAE_celeba VAE_gtsrb VAE_cifar VAE_imagenet

