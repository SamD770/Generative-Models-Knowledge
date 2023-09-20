
method_1=gradients_L2_norms_gaussian
method_2=typicality

for bs in 1 5
do
  python -m plots.anomaly_table.compare_methods \
    glow cifar_long celeba svhn_working imagenet32 gtsrb_glow_continued \
    --id_datasets cifar10 celeba svhn imagenet32 gtsrb \
    --datasets cifar10 celeba svhn imagenet32 gtsrb \
    --anomaly_detection $method_1 --batch_size $bs \
    --compare_to $method_2
done