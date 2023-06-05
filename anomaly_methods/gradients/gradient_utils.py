import torch

from copy import copy

from torch.utils.data import DataLoader

from data.utils import get_vanilla_dataset


from .gradient_serialisation import get_save_file_name
from path_definitions import GRADIENTS_DIR


def get_single_gradients(batch_size, model_name, id_dataset, ood_datasets):
    id_grads = torch.load(
        GRADIENTS_DIR
        + get_save_file_name(model_name, id_dataset, batch_size, method="single_grad")
    )

    ood_grads_list = [
        torch.load(
            GRADIENTS_DIR
            + get_save_file_name(
                model_name, ood_dataset, batch_size, method="single_grad"
            )
        )
        for ood_dataset in ood_datasets
    ]

    return id_grads, ood_grads_list


def get_raw_image_data(batch_size, dataset_name):
    dataset = get_vanilla_dataset(dataset_name)
    dl = DataLoader(dataset, batch_size=batch_size)

    return torch.stack([x.mean(0).flatten() for x, y in dl])


def get_norm_dict(
    batch_size, model_name, dataset_name, test_dataset=True, printout=False
):
    norm_file_name = get_save_file_name(
        model_name, dataset_name, batch_size, test_dataset=test_dataset
    )
    norms = torch.load(GRADIENTS_DIR + norm_file_name)

    zero_count = 0
    zero_keys = set()

    for key, value in norms.items():
        zeroes = torch.zeros(len(value))
        if torch.any(value == zeroes):
            zero_keys.add(key)
            zero_count += 1
    if printout:
        print(
            f"({dataset_name}, test_dataset={test_dataset}) number of zero gradients: {zero_count}"
        )

    return norms, zero_keys


def get_norms(
    batch_size,
    model_name,
    id_dataset,
    ood_datasets,
    printout=False,
    include_train=False,
):
    id_test_norms, id_zero_keys = get_norm_dict(
        batch_size, model_name, id_dataset, printout=printout
    )

    ood_norms_list = []
    all_zero_keys = id_zero_keys

    for ood_dataset in ood_datasets:
        norms, zero_keys = get_norm_dict(
            batch_size, model_name, ood_dataset, printout=printout
        )
        ood_norms_list.append(norms)
        all_zero_keys = all_zero_keys.union(zero_keys)

    all_norms = copy(ood_norms_list)
    all_norms.append(id_test_norms)

    if include_train:
        id_train_norms, zero_keys = get_norm_dict(
            batch_size, model_name, id_dataset, test_dataset=False, printout=printout
        )
        all_norms.append(id_train_norms)
        all_zero_keys.union(zero_keys)

    for key in all_zero_keys:
        for norms in all_norms:
            norms.pop(key)

    if include_train:
        return id_test_norms, id_train_norms, ood_norms_list
    else:
        return id_test_norms, ood_norms_list


def get_norms_depricated(
    batch_size, model_name, id_dataset, ood_datasets, printout=False
):
    id_norm_file = get_save_file_name(model_name, id_dataset, batch_size)

    ood_norm_files = [
        get_save_file_name(model_name, dataset_name, batch_size)
        for dataset_name in ood_datasets
    ]

    id_norms = torch.load(GRADIENTS_DIR + id_norm_file)

    ood_norms_list = [
        torch.load(GRADIENTS_DIR + ood_file) for ood_file in ood_norm_files
    ]

    all_norms = copy(ood_norms_list)
    all_norms.append(id_norms)

    all_names = copy(ood_datasets)
    all_names.append(id_dataset)

    zero_keys = set()

    for norms, name in zip(all_norms, all_names):
        zero_count = 0
        for key, value in norms.items():
            zeroes = torch.zeros(len(value))
            if torch.any(value == zeroes):
                zero_keys.add(key)
                zero_count += 1
        if printout:
            print(f"({name}) number of zero gradients: {zero_count}")

    if printout:
        print(f"removing {len(zero_keys)} gradients due to zero values")
    for key in zero_keys:
        for norms in all_norms:
            norms.pop(key)

    return id_norms, ood_norms_list


def get_stacked(norm_dict):
    return torch.stack(list(norm_dict.values()))


def get_norm_matrix(norm_dict):
    stacked = get_stacked(norm_dict)
    return torch.transpose(stacked, 0, 1)


def load_sklearn_norms(
    batch_size, model_name, id_dataset, ood_datasets, printout=False
):
    id_test_norms, id_train_norms, ood_norms_list = get_norms(
        batch_size,
        model_name,
        id_dataset,
        ood_datasets,
        printout=printout,
        include_train=True,
    )

    id_train_matrix = get_norm_matrix(id_train_norms)

    mu = torch.mean(id_train_matrix, 0)
    sigma = torch.std(id_train_matrix, 0)

    id_train_matrix = (id_train_matrix - mu) / sigma

    id_test_matrix = (get_norm_matrix(id_test_norms) - mu) / sigma

    ood_matrix_list = [
        (get_norm_matrix(norms) - mu) / sigma for norms in ood_norms_list
    ]

    return id_test_matrix, id_train_matrix, ood_matrix_list


def get_sklearn_norms(id_norms, ood_norms_list):
    """DEPRECATED: use load_sklearn_norms
    Returns tensors of norms ready for anomaly_methods using sklearn."""
    stacked_id_norms = get_stacked(id_norms)
    stacked_id_norms = torch.transpose(stacked_id_norms, 0, 1)

    id_mu = torch.mean(stacked_id_norms, 0)
    id_sigmas = torch.std(stacked_id_norms, 0)

    normed_id_norms = (stacked_id_norms - id_mu) / id_sigmas

    stacked_ood_norms_list = [
        torch.nan_to_num(  # Sometimes Nans appear in the input so this replaces them.
            torch.transpose(get_stacked(ood_norms), 0, 1)
        )
        for ood_norms in ood_norms_list
    ]

    normed_ood_norms_list = [
        (stacked_ood_norms - id_mu) / id_sigmas
        for stacked_ood_norms in stacked_ood_norms_list
    ]

    return normed_id_norms, normed_ood_norms_list


def split_id_data(id_data, fit_sample_proportion):
    id_samples, layer_count = id_data.shape
    fit_samples = round(id_samples * fit_sample_proportion)

    id_fit = id_data[:fit_samples]
    id_test = id_data[fit_samples:]

    return id_fit, id_test, fit_samples
