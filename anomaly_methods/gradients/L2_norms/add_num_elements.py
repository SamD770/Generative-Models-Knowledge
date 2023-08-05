"""Originally the summary statistic files for the L2 norms didn't contain data about the number of parameters."""

from models.utils import load_generative_model
from anomaly_methods.gradients.L2_norms import L2NormAnomalyDetection
import torch

# TODO: abstract this to an "everything generator"

for model_type, model_name_list in [
        # ("glow", ["cifar_long", "celeba", "svhn_working", "imagenet32"]),
        ("vae", ["VAE_cifar", "VAE_celeba", "VAE_svhn", "VAE_imagenet"]),
        ("diffusion", ["diffusion_cifar10", "diffusion_celeba", "diffusion_svhn", "diffusion_imagenet32"])
]:
    for model_name in model_name_list:

        model = load_generative_model(model_type, model_name)
        numel_dict = {
            name: p.numel() for name, p in model.named_parameters()
        }

        for dataset_name in ["cifar10", "svhn", "celeba", "imagenet32"]:
            for batch_size in [1, 5]:
                summary_statistic_filename = L2NormAnomalyDetection.summary_statistic_filepath(
                    model_type, model_name, "eval", dataset_name, batch_size
                )

                summary_stats = torch.load(summary_statistic_filename)
                new_summary_stats = {}

                first_key = iter(summary_stats.keys()).__next__()

                print(first_key)

                if type(first_key) is tuple:
                    continue

                for key, val in summary_stats.items():

                    # The new key is the pair of the parameter name and the number of elements.
                    new_key = (key, numel_dict[key])
                    new_summary_stats[new_key] = val

                torch.save(new_summary_stats, summary_statistic_filename)

