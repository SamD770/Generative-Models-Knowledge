bs=5
T=64

python -m plots.summary_statistic_histograms.entropy_differences \
      diffusion \
        diffusion_svhn_${T}_timesteps \
        diffusion_celeba_${T}_timesteps \
        diffusion_gtsrb_${T}_timesteps \
        diffusion_cifar10_${T}_timesteps \
      --id_datasets svhn celeba gtsrb cifar10 \
      --datasets svhn celeba gtsrb cifar10 \
      --batch_size $bs \
    --with_train_dataset_labels

T=512

python -m plots.summary_statistic_histograms.entropy_differences \
    diffusion \
      diffusion_svhn_${T}_timesteps \
      diffusion_celeba_${T}_timesteps \
      diffusion_gtsrb_${T}_timesteps \
      diffusion_cifar10_${T}_timesteps \
    --id_datasets svhn celeba gtsrb cifar10 \
    --datasets svhn celeba gtsrb cifar10 \
    --batch_size $bs \
    --with_legend



method_1=typicality
method_2=gradients_L2_norms_gaussian

for T in 1 2 4 8 16 32 64 128 256 512 999
do
  echo "\subsubsection{${T} timesteps}"
  echo
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


# Quick check to see performance using multiple samples
T=1
n=5
bs=5

echo "\subsubsection{${n} samples}"

python -m plots.anomaly_table.compare_methods \
  diffusion \
    diffusion_svhn_${T}_timesteps_${n}_samples \
    diffusion_celeba_${T}_timesteps_${n}_samples \
    diffusion_gtsrb_${T}_timesteps_${n}_samples \
    diffusion_cifar10_${T}_timesteps_${n}_samples \
    diffusion_imagenet32_${T}_timesteps_${n}_samples \
--id_datasets  svhn celeba gtsrb cifar10 imagenet32 \
--datasets svhn celeba gtsrb cifar10 imagenet32 \
--anomaly_detection $method_1 --batch_size $bs \
--compare_to $method_2
