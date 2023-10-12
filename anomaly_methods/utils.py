from anomaly_methods.gradients.L2_norms import (L2NormAnomalyDetection, OneClassSVML2Norm, DiagonalGaussianL2Norm,
                                                ChiSquareL2Norm, NaiveL2Norm, FisherMethodGaussianL2Norm)
from anomaly_methods.likelihoods import LikelihoodBasedAnomalyDetection, RawLikelihoodAnomalyDetection, TypicalityAnomalyDetection

anomaly_detection_methods_dict = {
    "gradients_L2_norms": L2NormAnomalyDetection,
    "gradients_L2_norms_naive": NaiveL2Norm,
    "gradients_L2_norms_svm": OneClassSVML2Norm,
    "gradients_L2_norms_gaussian": DiagonalGaussianL2Norm,
    "gradients_L2_norms_chi_square": ChiSquareL2Norm,
    "gradients_L2_norms_fishers_method": FisherMethodGaussianL2Norm,
    "likelihoods": LikelihoodBasedAnomalyDetection,
    "raw_likelihoods": RawLikelihoodAnomalyDetection,
    "typicality": TypicalityAnomalyDetection
}

anomaly_detection_methods = anomaly_detection_methods_dict.keys()


