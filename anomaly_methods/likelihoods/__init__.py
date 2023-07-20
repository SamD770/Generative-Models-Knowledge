"""
Contains the base classes for performing likelihood-based anomaly detection (currently using raw likelihoods and typicality)
"""
import torch

from path_definitions import LIKELIHOODS_DIR
from anomaly_methods.cached_statistic_filenames import get_save_file_name

from generative_model import GenerativeModel, AnomalyDetectionMethod
from typing import Dict, List, Optional

from torch import Tensor
from os import path


class LikelihoodBasedAnomalyDetection(AnomalyDetectionMethod):
    def __init__(self, summary_statistic_names, model: Optional[GenerativeModel] = None):
        super(LikelihoodBasedAnomalyDetection, self).__init__(summary_statistic_names, model)

    @staticmethod
    def get_summary_statistic_names(model) -> List[str]:
        return ["-log p"]

    def extract_summary_statistics(self, batch: Tensor) -> Dict[str, float]:
        """Takes the L^2 norm of each parameter's gradient and stores it in a dictionary."""
        with torch.no_grad():
            nlls = self.model.eval_nll(batch)
            joint_nll = nlls.sum().item()
            return {
                "-log p": joint_nll
            }

    @staticmethod
    def summary_statistic_filepath(model_name, dataset_name, batch_size):
        return path.join(LIKELIHOODS_DIR,
                         get_save_file_name(model_name, dataset_name, batch_size, method="likelihoods"))


class RawLikelihoodAnomalyDetection(LikelihoodBasedAnomalyDetection):
    def setup_method(self, fit_set_summary: Dict[str, List[float]]):
        pass

    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:
        return summary_statistics["-log p"]


class TypicalityAnomalyDetection(LikelihoodBasedAnomalyDetection):
    """
    Bootstrap implementation of method described in 'DETECTING OUT-OF-DISTRIBUTION INPUTS TO DEEP
    GENERATIVE MODELS USING TYPICALITY' Nalisnick et al. 2019
    """
    def __init__(self, summary_statistic_names, model: Optional[GenerativeModel] = None):
        super(LikelihoodBasedAnomalyDetection, self).__init__(summary_statistic_names, model)
        self.entropy_estimate = None

    def setup_method(self, fit_set_summary: Dict[str, List[float]]):

        # Computes the empirical estimate of the joint entropy H = E(log(p(x_1, x_2 ... x_B)))
        negative_log_likelihoods = fit_set_summary["-log p"]
        self.entropy_estimate = sum(negative_log_likelihoods)/len(negative_log_likelihoods)

    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:

        # Computes the distance from the joint log-likelihood to the entropy as per the cited paper

        negative_log_likelihoods = summary_statistics["-log p"]
        return [
            abs(nll - self.entropy_estimate) for nll in negative_log_likelihoods
        ]

