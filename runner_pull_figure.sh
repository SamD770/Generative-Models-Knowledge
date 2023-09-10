
method="likelihoods"
bs=1

python -m plots.summary_statistic_histograms \
    glow cifar_long \
    --id_datasets cifar10  \
    --datasets cifar10 svhn \
    --anomaly_detection $method --batch_size $bs


python -m plots.summary_statistic_histograms \
    glow svhn_working \
    --id_datasets svhn  \
    --datasets svhn cifar10 \
    --anomaly_detection $method --batch_size $bs