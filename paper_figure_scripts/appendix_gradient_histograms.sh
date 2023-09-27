method_name=gradients_L2_norms_gaussian
bs=5

# Canonical dataset ordering:

#    vae VAE_svhn VAE_celeba VAE_gtsrb VAE_cifar VAE_imagenet  \
#    glow svhn_working celeba gtsrb_glow_continued cifar_long imagenet32 \

#    diffusion \
#      diffusion_svhn_${T}_timesteps \
#      diffusion_celeba_${T}_timesteps \
#      diffusion_gtsrb_${T}_timesteps \
#      diffusion_cifar10_${T}_timesteps \
#      diffusion_imagenet32_${T}_timesteps \


#    --id_datasets  svhn celeba gtsrb cifar10 imagenet32 \
#    --datasets svhn celeba gtsrb cifar10 imagenet32 \



python -m plots.summary_statistic_histograms \
    vae VAE_svhn VAE_celeba VAE_gtsrb VAE_cifar VAE_imagenet  \
    --id_datasets  svhn celeba gtsrb cifar10 imagenet32 \
    --datasets svhn celeba gtsrb cifar10 imagenet32 \
    --anomaly_detection gradients_L2_norms_gaussian --batch_size $bs \
    --fitted_distribution \
    --n_statistics 4 --same_figure


T=1
python -m plots.summary_statistic_histograms \
    diffusion \
      diffusion_svhn_${T}_timesteps \
      diffusion_celeba_${T}_timesteps \
      diffusion_gtsrb_${T}_timesteps \
      diffusion_cifar10_${T}_timesteps \
      diffusion_imagenet32_${T}_timesteps \
    --id_datasets  svhn celeba gtsrb cifar10 imagenet32 \
    --datasets svhn celeba gtsrb cifar10 imagenet32 \
    --anomaly_detection gradients_L2_norms_gaussian --batch_size $bs \
    --fitted_distribution \
    --n_statistics 4 --same_figure


python -m plots.summary_statistic_histograms \
    glow svhn_working celeba gtsrb_glow_continued cifar_long imagenet32 \
    --id_datasets  svhn celeba gtsrb cifar10 imagenet32 \
    --datasets svhn celeba gtsrb cifar10 imagenet32 \
    --anomaly_detection gradients_L2_norms_gaussian --batch_size $bs \
    --fitted_distribution \
    --n_statistics 4 --same_figure

