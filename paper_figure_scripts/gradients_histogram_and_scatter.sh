method_name=gradients_L2_norms_gaussian
bs=5

python -m plots.summary_statistic_histograms.with_scatter \
    glow svhn_working celeba gtsrb_glow_continued cifar_long imagenet32 \
    --id_datasets  svhn celeba gtsrb cifar10 imagenet32 \
    --datasets svhn celeba gtsrb cifar10 imagenet32 \
    --anomaly_detection $method_name --batch_size $bs \
    --fitted_distribution \
    --with_legend

T=1

python -m plots.summary_statistic_histograms.with_scatter \
    diffusion \
      diffusion_svhn_${T}_timesteps \
      diffusion_celeba_${T}_timesteps \
      diffusion_gtsrb_${T}_timesteps \
      diffusion_cifar10_${T}_timesteps \
      diffusion_imagenet32_${T}_timesteps \
    --id_datasets  svhn celeba gtsrb cifar10 imagenet32 \
    --datasets svhn celeba gtsrb cifar10 imagenet32 \
  --anomaly_detection $method_name --batch_size $bs \
  --fitted_distribution

