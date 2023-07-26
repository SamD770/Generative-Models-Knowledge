def get_save_file_name(
    model_name,
    dataset_name,
    batch_size,
    model_mode="eval",
    method="norms",
    test_dataset=True,
    filetype="pt",
):

    if model_mode == "eval":
        model_mode_ext = "" # For backwards compatibility with pre-run files
    elif model_mode == "train":
        model_mode_ext = "_train_mode"
    else:
        model_mode_ext=  model_mode


    if test_dataset:
        file_name = (
            f"trained_{model_name}{model_mode_ext}_{method}_{dataset_name}_{batch_size}.{filetype}"
        )
    else:
        file_name = f"trained_{model_name}{model_mode_ext}_{method}_{dataset_name}-train_{batch_size}.{filetype}"
    return file_name
