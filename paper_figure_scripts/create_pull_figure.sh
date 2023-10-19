

python -m plots.summary_statistic_histograms.scatter_main\
    glow cifar_long \
    --id_datasets cifar10 \
    --datasets cifar10 svhn \
    --anomaly_detection "gradients_L2_norms" --batch_size 1
