## Gradients for anomaly detection

This code is that used for 
[On Gradients of Deep Generative Models for Representation-Invariant Anomaly Detection](https://openreview.net/forum?id=deYF9kVmIX)

- `generative_model.py` provides interfaces for general use of deep generative models for anomaly detection.
- `models/` contains several open source implementations of deep generative models
- `anomaly_methods/` contains implementations of deep-generative-model based anomaly detection
- `plots/` contains the general plotting scripts

```
@inproceedings{gradients2023anomaly,
  title={On Gradients of Deep Generative Models for Representation-Invariant Anomaly Detection},
  author={Sam Dauncey and Christopher C. Holmes and Christopher Williams and Fabian Falck},
  booktitle={ICLR 2023 Workshop on Pitfalls of limited data and computation for Trustworthy ML},
  url={https://openreview.net/forum?id=deYF9kVmIX}
  year={2023}
}
```

### Producing figures

The bash scripts used to produce the figures used in the paper are in `paper_figure_scripts/`, these call the python 
scripts in `plots/` with specific arguments.

#### Example: creating Figure 2

To replicate Figure 2 (using a randomly selected layer) run:

```angular2html
python -m plots.summary_statistic_histograms \ 
    glow [name_of_glow_model_trained_on_celeba] --id_datasets celeba \
    --datasets cifar10 celeba svhn imagenet32 gtsrb \
    --anomaly_detection gradients_L2_norms_gaussian --batch_size 5 \
    --fitted_distribution \
    --n_statistics 1
```

To run for all layers, remove the `n_statistics` flag.
