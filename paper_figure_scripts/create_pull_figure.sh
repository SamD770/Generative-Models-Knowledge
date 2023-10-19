
# TODO: DELETE, NOT USED IN PAPER

#
#python -m plots.summary_statistic_histograms \
#    glow cifar_long svhn_working \
#    --id_datasets cifar10 svhn \
#    --datasets cifar10 svhn \
#    --anomaly_detection $method --batch_size $bs \
#    --x_lim 0.0 2.0


python -m plots.summary_statistic_histograms.scatter_main\
    glow cifar_long \
    --id_datasets cifar10 \
    --datasets cifar10 svhn \
    --anomaly_detection "gradients_L2_norms" --batch_size 1


#
#python -m plots.summary_statistic_histograms.anomaly_score \
#    glow cifar_long \
#    --id_datasets cifar10 \
#    --datasets cifar10 svhn \
#    --anomaly_detection "gradients_L2_norms_naive" --batch_size 1
