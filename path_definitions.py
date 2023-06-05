

from os import path
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

MODELS_DIR = path.join(PROJECT_ROOT, "models")

VAE_ROOT = path.join(MODELS_DIR, "VAE_model")
GLOW_ROOT = path.join(MODELS_DIR, "glow_model")
PIXEL_CNN_ROOT = path.join(MODELS_DIR, "pixelCNN_model")
DIFFUSION_ROOT = path.join(MODELS_DIR, "diffusion_model")

GRADIENTS_DIR = path.join(PROJECT_ROOT, "anomaly_methods", "gradients", "serialised_gradients")
PLOTS_DIR = path.join(PROJECT_ROOT, "plots")

# GRADIENTS_DIR = "./anomaly_methods/gradients/serialised_gradients/"
