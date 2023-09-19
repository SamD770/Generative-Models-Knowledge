
for bs in 5 1
do
  for method in gradients_L2_norms_gaussian typicality
  do
    python -m plots.anomaly_table \
      glow cifar_long celeba svhn_working imagenet32 gtsrb_glow_continued \
      --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
      --datasets cifar10 celeba svhn imagenet32 gtsrb \
      --anomaly_detection $method --batch_size $bs
  done
done