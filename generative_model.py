"""
Provides standard interfaces for performing model-agnostic anomaly detection..
"""

import torch
import pickle

from torch import Tensor, Module
from torch.utils.data import Dataset, DataLoader

from typing import Dict, List


class GenerativeModel(Module):
    """Provides a standard interface for anomaly_methods code to interact with all types of models."""

    def eval_nll(self, x):
        raise NotImplementedError()

    def generate_sample(self, batch_size):
        raise NotImplementedError()

    @staticmethod
    def load_serialised(model_name):
        raise NotImplementedError()


class AnomalyDetectionMethod:
    """
    Provides a standard interface for performing anomaly detection using deep generative models. Each method provides
    a mapping of (model, data batch) -> dictionary of summary statistics (eg the norm of the gradient of each layer),
    the likelihood, etc.

    The pipeline for anomaly detection is then as follows:
    - compute_summary_statistics and cache_summary_statistics for each dataset
    - setup_method using the summary statistics for the in-distribution validation dataset
    - compute the anomaly_scores using the summary statistics for each dataset
    """

    def __init__(self, model: GenerativeModel):
        self.model = model
        self.summary_stats = {}

    @staticmethod
    def summary_statistic_names(model: GenerativeModel) -> List[str]:
        raise NotImplementedError()

    def extract_summary_statistics(self, batch: Tensor) -> Dict[str, float]:
        raise NotImplementedError()

    def setup_method(self, valid_set_summary: Dict[str, List[float]]):
        raise NotImplementedError()

    @staticmethod
    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:
        raise NotImplementedError()

    def compute_summary_statistics(self, dataset: Dataset, batch_size: int):
        """
        Applies extract_summary_statistics to each batch in dataset and stores the results in self.summary_stats
        :param dataset:
        :param batch_size:
        :return:
        """
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

        summary_stat_names = self.summary_statistic_names(self.model)

        self.summary_stats = {
            name: [] for name in summary_stat_names
        }

        for batch in dataloader:
            batch_summary_stats = self.extract_summary_statistics(batch)

            for name in summary_stat_names:
                self.summary_stats[name].append(
                    batch_summary_stats[name]
                )

    def cache_summary_statistics(self, filename, save_dir):
        pass # TODO: naming system to make sure that no





