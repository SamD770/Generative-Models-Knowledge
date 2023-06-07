import torch

from data.utils import device
from models.glow_model.model import Glow
from models.pixelCNN_model.main import PixelCNN
from models.VAE_model.model import SimpleVAE


model_class_dict = {
    "glow": Glow,
    "PixelCNN": PixelCNN,
    "vae": SimpleVAE
}


def load_generative_model(model_type, save_file, **params):
    model_class = {
        "glow": Glow,
        "PixelCNN": PixelCNN,
        "vae": SimpleVAE
    }[model_type]

    return model_class.load_serialised(save_file, **params)


def compute_nll(dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=1)

    nlls = []
    for x, y in dataloader:
        x = x.to(device)

        # if hparams['y_condition']:
        #     y = y.to(device)
        # else:
        #     y = None

        with torch.no_grad():
            nll = model.eval_nll(x)
            nlls.append(nll)

    return torch.cat(nlls).cpu()
