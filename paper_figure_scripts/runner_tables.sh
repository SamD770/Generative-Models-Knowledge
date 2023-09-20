
method_1=typicality
method_2=gradients_L2_norms_gaussian

for bs in 1 5
do
  python -m plots.anomaly_table.compare_methods \
    glow cifar_long celeba svhn_working imagenet32 gtsrb_glow_continued \
    --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
    --datasets cifar10 celeba svhn imagenet32 gtsrb \
    --anomaly_detection $method_1 --batch_size $bs \
    --compare_to $method_2
done

T=1

for bs in 1 5
do
  python -m plots.anomaly_table.compare_methods \
    diffusion \
      diffusion_cifar10_${T}_timesteps \
      diffusion_celeba_${T}_timesteps \
      diffusion_svhn_${T}_timesteps \
      diffusion_imagenet32_${T}_timesteps \
      diffusion_gtsrb_${T}_timesteps \
    --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
    --datasets cifar10 celeba svhn imagenet32 gtsrb \
    --anomaly_detection $method_1 --batch_size $bs \
    --compare_to $method_2
done