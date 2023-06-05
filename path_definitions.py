

from os import path
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

VAE_ROOT = path.join(PROJECT_ROOT, "models/VAE_model")
GLOW_ROOT = path.join(PROJECT_ROOT, "models/glow_model")
PIXEL_CNN_ROOT = path.join(PROJECT_ROOT, "models/pixelCNN_model")
DIFFUSION_ROOT = path.join(PROJECT_ROOT, "models/diffusion_model")

GRADIENTS_DIR = path.join(PROJECT_ROOT, "anomaly_methods", "gradients", "serialised_gradients")

# GRADIENTS_DIR = "./anomaly_methods/gradients/serialised_gradients/"
