method_name=gradients_L2_norms_gaussian
bs=5

python -m plots.summary_statistic_histograms.with_scatter \
    glow cifar_long celeba svhn_working imagenet32 gtsrb_glow_continued \
    --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
    --datasets cifar10 celeba svhn imagenet32 gtsrb \
    --anomaly_detection $method_name --batch_size $bs \
    --fitted_distribution

T=1

python -m plots.summary_statistic_histograms.with_scatter \
    diffusion \
      diffusion_cifar10_${T}_timesteps \
      diffusion_celeba_${T}_timesteps \
      diffusion_svhn_${T}_timesteps \
      diffusion_imagenet32_${T}_timesteps \
      diffusion_gtsrb_${T}_timesteps \
  --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
  --datasets cifar10 celeba svhn imagenet32 gtsrb \
  --anomaly_detection $method_name --batch_size $bs \
  --fitted_distribution

