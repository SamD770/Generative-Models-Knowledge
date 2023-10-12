method_1=gradients_L2_norms_fishers_method
method_2=gradients_L2_norms_gaussian

for bs in 1 5
do
  python -m plots.anomaly_table.compare_methods \
    glow svhn_working celeba gtsrb_glow_continued cifar_long imagenet32 \
    --id_datasets  svhn celeba gtsrb cifar10 imagenet32 \
    --datasets svhn celeba gtsrb cifar10 imagenet32 \
    --anomaly_detection $method_1 --batch_size $bs \
    --compare_to $method_2
done

T=1

for bs in 1 5
do
  python -m plots.anomaly_table.compare_methods \
    diffusion \
      diffusion_svhn_${T}_timesteps \
      diffusion_celeba_${T}_timesteps \
      diffusion_gtsrb_${T}_timesteps \
      diffusion_cifar10_${T}_timesteps \
      diffusion_imagenet32_${T}_timesteps \
    --id_datasets  svhn celeba gtsrb cifar10 imagenet32 \
    --datasets svhn celeba gtsrb cifar10 imagenet32 \
    --anomaly_detection $method_1 --batch_size $bs \
    --compare_to $method_2
done
