
# TODO: DELETE, NOT USED IN PAPER

method="typicality"
bs=1

python -m plots.summary_statistic_histograms \
    glow cifar_long svhn_working \
    --id_datasets cifar10 svhn \
    --datasets cifar10 svhn \
    --anomaly_detection $method --batch_size $bs \
    --x_lim 0.0 2.0


python -m plots.anomaly_score_histograms \
    glow cifar_long svhn_working \
    --id_datasets cifar10 svhn \
    --datasets cifar10 svhn \
    --anomaly_detection $method --batch_size $bs
