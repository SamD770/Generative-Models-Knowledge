"""
Contains the base classes for performing likelihood-based anomaly detection (currently using raw likelihoods and typicality)
"""
from generative_model import GenerativeModel, AnomalyDetectionMethod


class LikelihoodBasedAnomalyDetection(AnomalyDetectionMethod):
    pass


class RawLikelihoodAnomalyDetection(LikelihoodBasedAnomalyDetection):
    pass


class TypicalityAnomalyDetection(LikelihoodBasedAnomalyDetection):
    pass
