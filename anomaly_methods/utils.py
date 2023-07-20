from anomaly_methods.gradients.L2_norms import L2NormAnomalyDetection, OneClassSVML2Norm, DiagonalGaussianL2Norm
from anomaly_methods.likelihoods import LikelihoodBasedAnomalyDetection, RawLikelihoodAnomalyDetection, TypicalityAnomalyDetection

anomaly_detection_methods_dict = {
    "gradients_L2_norms": L2NormAnomalyDetection,
    "gradients_L2_norms_svm": OneClassSVML2Norm,
    "gradients_L2_norms_gaussian": DiagonalGaussianL2Norm,
    "likelihoods": LikelihoodBasedAnomalyDetection,
    "raw_likelihoods": RawLikelihoodAnomalyDetection,
    "typicality": TypicalityAnomalyDetection
}

anomaly_detection_methods = anomaly_detection_methods_dict.keys()


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
