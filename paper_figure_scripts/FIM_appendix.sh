
python -m plots.FIM_approximation.raw_FIMs \
  glow svhn_working celeba gtsrb_glow_continued cifar_long imagenet32 \

python -m plots.FIM_approximation.raw_FIMs diffusion \
    diffusion_svhn_1_timesteps \
    diffusion_celeba_1_timesteps \
    diffusion_gtsrb_1_timesteps \
    diffusion_cifar10_1_timesteps \
    diffusion_imagenet32_1_timesteps