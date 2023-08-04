"""Originally the summary statistic files for the L2 norms didn't contain data about the number of parameters."""

from models.utils import load_generative_model
from anomaly_methods.gradients.L2_norms import L2NormAnomalyDetection
import torch

# TODO: abstract this to an "everything generator"

for model_type, model_name_list in [("glow", ["svhn_working"])]:
    for model_name in model_name_list:

        model = load_generative_model(model_type, model_name)
        numel_dict = {
            name: p.numel() for name, p in model.named_parameters()
        }

        for dataset_name in ["cifar"]:
            for batch_size in [1, 5]:
                summary_statistic_filename = L2NormAnomalyDetection.summary_statistic_filepath(
                    model_type, model_name, "eval", dataset_name, batch_size
                )

                summary_stats = torch.load(summary_statistic_filename)
                new_summary_stats = {}

                for key, val in summary_stats.items():

                    # The new key is the pair of the parameter name and the number of elements.
                    new_key = (key, numel_dict[key])
                    print(new_key)
                    print(val)
                    exit()
                    new_summary_stats[new_key] = val
