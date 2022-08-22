from gradient_utils import *

from scipy.stats import kstest


def do_ks_test(id_data, ood_data_list, fit_sample_proportion=0.8):
    id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates = None, None, None

    ood_rejection_rates = []
    for ood_data in ood_data_list:
        ks_stat, p_val = kstest(id_data, ood_data)
        ood_rejection_rates.append(p_val)

    return id_fit_rejection_rate, id_test_rejection_rate, ood_rejection_rates


if __name__ == "__main__":
    model_name = "cifar_long"
    id_dataset = "cifar"
    dataset_names = ["svhn", "celeba", "imagenet32"]

    id_grads, ood_grads_list = get_single_gradients(1, model_name, id_dataset, dataset_names)
    _, _, p_vals = do_ks_test(id_grads, ood_grads_list)

    for dataset_name, p_val in zip(dataset_names, p_vals):
        print("dataset name:", dataset_name)
        print("p value:", p_val)