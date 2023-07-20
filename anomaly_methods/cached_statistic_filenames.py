def get_save_file_name(
    model_name,
    dataset_name,
    batch_size,
    method="norms",
    test_dataset=True,
    filetype="pt",
):
    if test_dataset:
        file_name = (
            f"trained_{model_name}_{method}_{dataset_name}_{batch_size}.{filetype}"
        )
    else:
        file_name = f"trained_{model_name}_{method}_{dataset_name}-train_{batch_size}.{filetype}"
    return file_name
