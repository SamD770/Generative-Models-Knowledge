method_name=gradients_L2_norms_gaussian
bs=5


python -m plots.summary_statistic_histograms \
    vae VAE_cifar VAE_celeba VAE_svhn VAE_imagenet VAE_gtsrb \
    --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
    --datasets cifar10 celeba svhn imagenet32 gtsrb \
    --anomaly_detection gradients_L2_norms_gaussian --batch_size $bs \
    --fitted_distribution \
    --n_statistics 4 --same_figure


T=1
python -m plots.summary_statistic_histograms \
    diffusion \
      diffusion_cifar10_${T}_timesteps \
      diffusion_celeba_${T}_timesteps \
      diffusion_svhn_${T}_timesteps \
      diffusion_imagenet32_${T}_timesteps \
      diffusion_gtsrb_${T}_timesteps \
    --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
    --datasets cifar10 celeba svhn imagenet32 gtsrb \
    --anomaly_detection gradients_L2_norms_gaussian --batch_size $bs \
    --fitted_distribution \
    --n_statistics 4 --same_figure


python -m plots.summary_statistic_histograms \
    glow cifar_long celeba svhn_working imagenet32 gtsrb_glow_continued \
    --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
    --datasets cifar10 celeba svhn imagenet32 gtsrb \
    --anomaly_detection gradients_L2_norms_gaussian --batch_size $bs \
    --fitted_distribution \
    --n_statistics 4 --same_figure

