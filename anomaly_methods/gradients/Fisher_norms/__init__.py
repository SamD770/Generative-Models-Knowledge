from anomaly_methods.cached_statistic_filenames import get_save_file_name
from anomaly_methods.gradients.gradient_utils import backprop_nll
from path_definitions import FISHER_NORMS_DIR

from os import path

from generative_model import AnomalyDetectionMethod, GenerativeModel
from typing import Dict, List, Optional

from torch import Tensor


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
    def get_summary_statistic_names(model) -> List[str]:
        return [
            "fisher_norm"
        ]

    def extract_summary_statistics(self, batch: Tensor) -> Dict[str, float]:
        # TODO: If FIM uncomputed, iterate until convergence to compute it. Then use it.
        pass

    @staticmethod
    def summary_statistic_filepath(model_type, model_name, dataset_name, batch_size):
        return path.join(FISHER_NORMS_DIR, model_type,
                         get_save_file_name(model_name, dataset_name, batch_size))

    def setup_method(self, fit_set_summary: Dict[str, List[float]]):
        pass # Maybe store average gradient vector

    def anomaly_score(self, summary_statistics: Dict[str, List[float]]) -> List[float]:
        pass
