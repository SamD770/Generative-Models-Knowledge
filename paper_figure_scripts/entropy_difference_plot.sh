bs=1

python -m plots.summary_statistic_histograms.entropy_differences \
    glow svhn_working celeba gtsrb_glow_continued cifar_long \
    --id_datasets  svhn celeba gtsrb cifar10 \
    --datasets svhn celeba gtsrb cifar10 \
    --batch_size $bs \
    --x_lim -6.5 -1.0 \
    --with_train_dataset_labels

T=1

python -m plots.summary_statistic_histograms.entropy_differences \
    diffusion \
      diffusion_svhn_${T}_timesteps \
      diffusion_celeba_${T}_timesteps \
      diffusion_gtsrb_${T}_timesteps \
      diffusion_cifar10_${T}_timesteps \
    --id_datasets  svhn celeba gtsrb cifar10 \
    --datasets svhn celeba gtsrb cifar10 \
    --batch_size $bs \
    --x_lim -1.0 -0.3 \
    --with_legend