from anomaly_methods.cached_statistic_filenames import get_save_file_name
from anomaly_methods.gradients.gradient_utils import backprop_nll
from path_definitions import L2_NORMS_DIR

from generative_model import AnomalyDetectionMethod, GenerativeModel
from typing import Dict, List, Optional
from os import path

import torch
from torch import Tensor
from torch.distributions.normal import Normal

from sklearn.svm import OneClassSVM


def zero_keys(summary_stats, printout=True):
    """Returns a set of keys for which the corresponding summary statistics have 0 standard deviation
    (Avoids ZeroDivisionErrors)."""
    zero_count = 0
    zero_keys = set()

    for key, value in summary_stats.items():

        if torch.std(value) == 0:
            zero_keys.add(key)
            zero_count += 1

    if printout:
        print(
            f"number of zero gradients: {zero_count}"
        )

    return zero_keys


def get_stacked(summary_stats, zero_keys_valid):
    # Stacks the summary statistics into a matrix, skipping over those in zero_keys_valid
    return torch.stack([
        stat for stat_name, stat in summary_stats.items() if stat_name not in zero_keys_valid
    ])


def get_summary_matrix(summary_stats, zero_keys_valid):
    """Stacks the summary statistics so that they can be passed to a sklearn model."""
    stacked = get_stacked(summary_stats, zero_keys_valid)
    matrix = torch.transpose(stacked, 0, 1)
    return torch.nan_to_num(matrix)


# TODO: maybe override from_model

class L2NormAnomalyDetection(AnomalyDetectionMethod):
    """
    An AnomalyDetectionMethod that fits an IsolationForest (Liu et al.) to the norms of the gradient of each parameter.
    See https://openreview.net/forum?id=deYF9kVmIX for description of method.
    """
    def __init__(self, summary_statistic_names, model: Optional[GenerativeModel] = None):
        super(L2NormAnomalyDetection, self).__init__(summary_statistic_names, model)

        # We need to store the following information about the validation set when we run setup_method
        self.zero_keys_fit = set()
        self.mean_fit = None
        self.stdev_fit = None
        self.sklearn_model = None

    @staticmethod
    def get_summary_statistic_names(model) -> List:
        return [
            (name, p.numel()) for name, p in model.named_parameters()
        ]

    def extract_summary_statistics(self, batch: Tensor) -> Dict[str, float]:
        """Takes the L^2 norm of each parameter's gradient and stores it in a dictionary."""
        def take_norm(grad_vec):
            return (grad_vec ** 2).sum().item()

        backprop_nll(self.model, batch)

        return {
            (name, p.numel()): take_norm(p.grad) for name, p in self.model.named_parameters()
        }

    @staticmethod
    def summary_statistic_filepath(model_type, model_name, model_mode, dataset_name, batch_size):
        return path.join(L2_NORMS_DIR,
                         get_save_file_name(model_name, dataset_name, batch_size, model_mode=model_mode))


class SKLearnL2NormAnomalyDetection(L2NormAnomalyDetection):
    def setup_method(self, fit_set_summary: Dict[str, List[float]]):
        # Pre-treats the raw statistics to make them suitable for the sklearn
        self.zero_keys_fit = zero_keys(fit_set_summary)

        summary_matrix_fit = get_summary_matrix(fit_set_summary, self.zero_keys_fit)

        self.mean_fit = torch.mean(summary_matrix_fit, 0)
        self.stdev_fit = torch.std(summary_matrix_fit, 0)

        normed_summary_fit = (summary_matrix_fit - self.mean_fit) / self.stdev_fit

        print(normed_summary_fit.isnan().sum())

        if normed_summary_fit.isnan().any():
            raise ValueError(f"summary matrix with shape {normed_summary_fit.shape} "
                             f"contains {normed_summary_fit.isnan().sum()} Nan values")

        # Fits the scikit-learn model

        self.sklearn_model = self.get_sklearn_model().fit(normed_summary_fit)

    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:
        summary_matrix = get_summary_matrix(summary_statistics, self.zero_keys_fit)
        normed_summary = (summary_matrix - self.mean_fit) / self.stdev_fit

        if normed_summary.isnan().any():
            raise ValueError(f"summary matrix contains {normed_summary.isnan().sum()} Nan values")

        return self.sklearn_model.score_samples(normed_summary)

    @classmethod
    def get_sklearn_model(cls):
        raise NotImplementedError()


class OneClassSVML2Norm(SKLearnL2NormAnomalyDetection):
    @classmethod
    def get_sklearn_model(cls):
        return OneClassSVM(nu=0.001)


# TODO: unify these bois under one flag.


class DiagonalGaussianL2Norm(L2NormAnomalyDetection):
    """
    Fits a Log-normal distribution to each feature, the anomaly score is the log-probability-density assuming independence.
    """

    def log_value_generator(self, summary_statistics):
        """Pre-treats the values in summary_statistics by converting to a tensor and taking logs."""
        for summary_stat_name, value_list in summary_statistics.items():
            if summary_stat_name in self.zero_keys_fit:
                continue

            value_tensor = torch.tensor(value_list)
            log_value_tensor = torch.log(value_tensor)
            yield summary_stat_name, log_value_tensor

    def setup_method(self, fit_set_summary: Dict[str, List[float]]):
        # Fits a log-normal distribution to each feature
        self.zero_keys_fit = zero_keys(fit_set_summary)

        self.normal_dists = {}

        for summary_stat_name, log_value_tensor in self.log_value_generator(fit_set_summary):

            normal_dist = Normal(log_value_tensor.mean(), log_value_tensor.std(dim=0))

            self.normal_dists[summary_stat_name] = normal_dist

    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:

        # Computes the log-probability of each sample in parallel.
        log_p = 0

        for summary_stat_name, log_value_tensor in self.log_value_generator(summary_statistics):

            try:
                log_value_tensor = log_value_tensor.nan_to_num()
                layer_model_density = self.normal_dists[summary_stat_name].log_prob(log_value_tensor)
                log_p += layer_model_density
            except ValueError:
                print(log_value_tensor.isnan().any())
                print(log_value_tensor)

        return list(
            val.item() for val in log_p
        )


class ChiSquareL2Norm(L2NormAnomalyDetection):
    """
    Fits a scaled Chi-Square distribution to the L2-norms, assuming independence
    """
    def setup_method(self, fit_set_summary: Dict[str, List[float]]):
        # Fits a chi-square distribution times by some scale to each feature
        self.zero_keys_fit = zero_keys(fit_set_summary)
        self.scales = {}

        for key, value in fit_set_summary.items():
            _, num_elements = key
            self.scales[key] = torch.tensor(value).mean() / num_elements # This is the MLE

    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:
        return sum(
            torch.tensor(value) * num_elements for (_, num_elements), value in summary_statistics.items()
        )
