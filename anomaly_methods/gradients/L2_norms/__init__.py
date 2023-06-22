from path_definitions import L2_NORMS_DIR

from generative_model import AnomalyDetectionMethod, GenerativeModel
from typing import Dict, List
from os import path

import torch
from torch import Tensor

from sklearn.ensemble import IsolationForest


def get_save_file_name(
    model_name,
    dataset_name,
    batch_size,
    method="norms",
    test_dataset=True,
    filetype="pt",
):
    if test_dataset:
        file_name = (
            f"trained_{model_name}_{method}_{dataset_name}_{batch_size}.{filetype}"
        )
    else:
        file_name = f"trained_{model_name}_{method}_{dataset_name}-train_{batch_size}.{filetype}"
    return file_name


def zero_keys(summary_stats, printout=False):
    """Returns a set of keys for which the corresponding summary statistics are 0 (Avoids ZeroDivisionErrors)."""
    zero_count = 0
    zero_keys = set()

    for key, value in summary_stats.items():
        zeroes = torch.zeros(len(value))
        if torch.any(value == zeroes):
            zero_keys.add(key)
            zero_count += 1
    if printout:
        print(
            f"number of zero gradients: {zero_count}"
        )

    return zero_keys


def get_stacked(summary_stats, zero_keys_valid):
    return torch.stack([
        stats for stats in summary_stats.values() if stats not in zero_keys_valid
    ])


def get_summary_matrix(summary_stats, zero_keys_valid):
    """Stacks the summary statistics so that they can be passed to a sklearn model."""
    stacked = get_stacked(summary_stats, zero_keys_valid)
    return torch.transpose(stacked, 0, 1)


class L2NormAnomalyDetection(AnomalyDetectionMethod):
    """
    An AnomalyDetectionMethod that fits an IsolationForest (Liu et al.) to the norms of the gradient of each parameter.
    See https://openreview.net/forum?id=deYF9kVmIX for description of method.
    """
    def __init__(self, model: GenerativeModel):
        super(L2NormAnomalyDetection, self).__init__(model)

        # We need to store the following information about the validation set when we run setup_method
        self.zero_keys_fit = set()
        self.mean_fit = None
        self.stdev_fit = None
        self.sklearn_model = None

    @staticmethod
    def summary_statistic_names(model: GenerativeModel) -> List[str]:
        return [
            name for name, _ in model.named_parameters()
        ]

    def extract_summary_statistics(self, batch: Tensor) -> Dict[str, float]:
        """Takes the L^2 norm of each parameter's gradient and stores it in a dictionary."""
        def take_norm(grad_vec):
            return (grad_vec ** 2).sum().item()

        return {
            name: take_norm(p.grad) for name, p in self.model.named_parameters()
        }

    def setup_method(self, fit_set_summary: Dict[str, List[float]]):
        # Fits pre-treats the raw statistics to make them suitable for the IsolationForest
        self.zero_keys_fit = zero_keys(fit_set_summary)

        summary_matrix_fit = get_summary_matrix(fit_set_summary, self.zero_keys_fit)

        self.mean_fit = torch.mean(summary_matrix_fit, 0)
        self.stdev_fit = torch.std(summary_matrix_fit, 0)

        normed_summary_fit = (summary_matrix_fit - self.mean_fit) / self.stdev_fit

        # Fits the IsolationForest

        self.sklearn_model = IsolationForest(n_estimators=10000).fit(normed_summary_fit)

    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:

        summary_matrix = get_summary_matrix(summary_statistics, self.zero_keys_fit)
        normed_summary = (summary_matrix - self.mean_fit) / self.stdev_fit

        return self.sklearn_model.score_samples(normed_summary)

    @staticmethod
    def summary_statistic_filepath(model_name, dataset_name, batch_size):
        return path.join(L2_NORMS_DIR,
                         get_save_file_name(model_name, dataset_name, batch_size))
