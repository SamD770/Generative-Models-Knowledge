

from os import path
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()

VAE_ROOT = path.join(PROJECT_ROOT, "VAE_model")
GLOW_ROOT = path.join(PROJECT_ROOT, "GLOW_model")
PIXEL_CNN_ROOT = path.join(PROJECT_ROOT, "PixelCNN_model")
DIFFUSION_ROOT = path.join(PROJECT_ROOT, "diffusion_model")



