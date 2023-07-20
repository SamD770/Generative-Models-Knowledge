"""
Contains the base classes for performing likelihood-based anomaly detection (currently using raw likelihoods and typicality)
"""
from path_definitions import LIKELIHOODS_DIR
from anomaly_methods.utils import get_save_file_name

from generative_model import GenerativeModel, AnomalyDetectionMethod
from typing import Dict, List, Optional

from torch import Tensor
from os import path


class LikelihoodBasedAnomalyDetection(AnomalyDetectionMethod):
    def __init__(self, summary_statistic_names, model: Optional[GenerativeModel] = None):
        super(LikelihoodBasedAnomalyDetection, self).__init__(summary_statistic_names, model)

        pass

    @staticmethod
    def get_summary_statistic_names(model) -> List[str]:
        return ["log p"]

    def extract_summary_statistics(self, batch: Tensor) -> Dict[str, float]:
        """Takes the L^2 norm of each parameter's gradient and stores it in a dictionary."""
        nlls = self.model.eval_nll(batch)
        mean_nll = nlls.mean()
        return {
            "log p": mean_nll
        }

    @staticmethod
    def summary_statistic_filepath(model_name, dataset_name, batch_size):
        return path.join(LIKELIHOODS_DIR,
                         get_save_file_name(model_name, dataset_name, batch_size))


class RawLikelihoodAnomalyDetection(LikelihoodBasedAnomalyDetection):
    def setup_method(self, fit_set_summary: Dict[str, List[float]]):
        pass

    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:
        return summary_statistics["log p"]


class TypicalityAnomalyDetection(LikelihoodBasedAnomalyDetection):
    pass
