from analysis import *

DEFAULT_SAMPLES = 512
BATCH_SIZE = 32


def monte_carlo_distribution(
    generating_model, model_list, hparam_list, function, samples=DEFAULT_SAMPLES
):
    """General function for computing monte carlo estimates for distribution statistics."""
    sample_dataset = SampleDataset(generating_model, batch_count=samples // BATCH_SIZE)
    nll_list = []
    for model, hparams in zip(model_list, hparam_list):
        nll_list.append(compute_nll(sample_dataset, model, hparams))
    estimates = function(nll_list)
    return torch.mean(estimates)


def monte_carlo_entropy(model, hparams, samples=DEFAULT_SAMPLES):
    """Performs a monte carlo estimate of the entropy of the distribution of model."""
    return monte_carlo_distribution(
        model, [model], [hparams], lambda nlls: nlls[0], samples=samples
    )


def monte_carlo_cross_entropy(
    model_1, model_2, model2_hparams, samples=DEFAULT_SAMPLES
):
    """Performs a monte carlo estimate of the cross-entropy between model1 and model2."""
    return monte_carlo_distribution(
        model_1, [model_2], [model2_hparams], lambda nlls: nlls[0], samples=samples
    )


def monte_carlo_KL(
    model_1, model_2, model_1_hparams, model_2_hparams, samples=DEFAULT_SAMPLES
):
    """Performs a monte carlo estimate of the KL divergence between model1 and model2"""
    return monte_carlo_distribution(
        model_1,
        [model_1, model_2],
        [model_1_hparams, model_2_hparams],
        lambda nlls_1, nlls_2: nlls_1 - nlls_2,
        samples=samples,
    )


svhn_model, svhn_hparams = load_glow_model("svhn_glow/", "glow_checkpoint_286000.pt")
cifar_model, cifar_hparams = load_glow_model("cifar_glow/", "glow_checkpoint_195250.pt")

print("models loaded")

print(f"svhn learned entropy: {monte_carlo_entropy(svhn_model, svhn_hparams)}")
print(f"cifar learned entropy: {monte_carlo_entropy(cifar_model, cifar_hparams)}")
