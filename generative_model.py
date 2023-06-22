"""
Provides standard interfaces for performing model-agnostic anomaly detection..
"""

import torch
import pickle

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from typing import Dict, List


class GenerativeModel:
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
    Provides a standard interface for performing anomaly detection using deep generative models.
    the pipeline for doing so is using the model and a batch of data to provide one dictionary of summary statistics.
    The summary statistics for data can be cached and subsequently loaded to generate anomaly scores
    """

    def __init__(self, model: GenerativeModel):
        self.model = model
        self.summary_stats = {}

    @staticmethod
    def summary_statistic_names(model: GenerativeModel) -> List[str]:
        raise NotImplementedError()

    def extract_summary_statistics(self, batch: Tensor) -> Dict[str, float]:
        raise NotImplementedError()

    def setup_method(self, train_set_summary: Dict[str, List[float]]):
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
        pass





