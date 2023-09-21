bs=1

python -m plots.summary_statistic_histograms.entropy_differences \
    glow cifar_long celeba svhn_working \
    --id_datasets cifar10 celeba svhn \
    --datasets cifar10 celeba svhn imagenet32 gtsrb \
    --batch_size $bs

#T=1
#
#python -m plots.summary_statistic_histograms.entropy_differences \
#    diffusion \
#      diffusion_cifar10_${T}_timesteps \
#      diffusion_celeba_${T}_timesteps \
#      diffusion_svhn_${T}_timesteps \
#    --id_datasets cifar10 celeba svhn \
#    --datasets cifar10 celeba svhn imagenet32 gtsrb \
#    --batch_size $bs