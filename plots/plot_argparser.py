import argparse
import plots

from models.glow_model.model import Glow
from models.pixelCNN_model.main import PixelCNN
from models.VAE_model.model import SimpleVAE


model_class_dict = {
    "glow": Glow,
    "PixelCNN": PixelCNN,
    "vae": SimpleVAE
}


def get_plot_argparser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model_class", choices=model_class_dict.keys())
    parser.add_argument("--model_save_file")


def load_generative_model(model_type, save_file, **params):

    model_class = model_class_dict[model_type]

    return model_class.load_serialised(save_file, **params)


def get_model(args):
    return load_generative_model(args.model_class, args.model_save_file)