
method="likelihoods"
bs=1

python -m plots.summary_statistic_histograms \
    glow cifar_long svhn_working \
    --id_datasets cifar10 svhn \
    --datasets cifar10 svhn \
    --anomaly_detection $method --batch_size $bs
