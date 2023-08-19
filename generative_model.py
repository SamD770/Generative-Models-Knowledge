"""
Provides standard interfaces for performing model-agnostic anomaly detection..
"""

import torch
import pickle
from datetime import datetime

from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from typing import Dict, List, Optional


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
    Provides a standard interface for performing anomaly detection using deep generative models. Each method provides
    a mapping of (model, data batch) -> dictionary of summary statistics (eg the norm of the gradient of each layer),
    the likelihood, etc.

    The pipeline for anomaly detection is then as follows:
    - compute_summary_statistics and cache_summary_statistics for each dataset
    - setup_method using the summary statistics for the in-distribution "fitting" dataset
    - compute the anomaly_scores using the summary statistics for each dataset

    anomaly_score should be higher for in-distribution data and lower for out-of-distribution data.
    """

    def __init__(self, summary_statistic_names, model: Optional[GenerativeModel] = None):
        self.summary_statistic_names = summary_statistic_names
        self.model = model

    @classmethod
    def from_model(cls, model):
        summary_statistic_names = cls.get_summary_statistic_names(model)
        return cls(summary_statistic_names, model)

    @classmethod
    def from_dataset_summary(cls, dataset_summary):
        return cls(dataset_summary.keys())

    @staticmethod
    def get_summary_statistic_names(model) -> List[str]:
        raise NotImplementedError()

    def extract_summary_statistics(self, batch: Tensor) -> Dict[str, float]:
        raise NotImplementedError()

    def setup_method(self, fit_set_summary: Dict[str, List[float]]):
        raise NotImplementedError()

    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:
        raise NotImplementedError()

    @staticmethod
    def summary_statistic_filepath(model_type, model_name, model_mode, dataset_name, batch_size):
        raise NotImplementedError()

    def dataset_summary_dict(self):
        return {
            name: [] for name in self.summary_statistic_names
        }

    def compute_summary_statistics(self, dataset: Dataset, batch_size: int, model_mode="eval", verbose=True):
        """
        Applies extract_summary_statistics to each batch in dataset
        :param dataset:
        :param batch_size:
        :return:
        """
        # This whole process could be parallelized; which would speed up the process especially for batch_size 1
        #   This could use call_for_per_sample_grad from PyTorch 2.0 when dealing with gradient-based methods.
        #   Or it could use torch.multiprocessing and no ./anomaly_methods/ code would need to be changed.
        #   Engineering time ~6 hours.

        if self.model is None:
            raise ValueError("Attempted to extract summary statistics without initialising a model.")

        self.model.to(device)

        if model_mode == "eval":
            self.model.eval()
        elif model_mode == "train":
            self.model.train()
        else:
            raise ValueError(f"Model mode {model_mode} not recognised.")

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

        dataset_summary = self.dataset_summary_dict()

        print_update_every = len(dataset) // (20 * batch_size)

        for i, (batch, labels) in enumerate(dataloader):
            batch = batch.to(device)

            batch_summary_stats = self.extract_summary_statistics(batch)

            for name in self.summary_statistic_names:
                dataset_summary[name].append(
                    batch_summary_stats[name]
                )

            if verbose and i % print_update_every == 0:
                print(f"({datetime.now()}) {i * batch_size}/{len(dataset)} complete")

        dataset_summary = {
            key: torch.tensor(value) for key, value in dataset_summary.items()
        }

        return dataset_summary

    def split_dataset_summary(self, dataset_summary, proportion):
        summary_1 = {}
        summary_2 = {}

        summary_length = None

        for name in self.summary_statistic_names:

            statistic_list = dataset_summary[name]
            statistic_count = len(statistic_list)

            if summary_length is None:
                summary_length = statistic_count
                split_count = round(summary_length*proportion)

            elif summary_length != statistic_count:
                raise ValueError(f"Tried to split a dataset summary with uneven lengths: {name} has length "
                                 f"{statistic_count} but this value was {summary_length} for other statistics.")

            summary_1[name] = statistic_list[:split_count]
            summary_2[name] = statistic_list[split_count:]

        return summary_1, summary_2




