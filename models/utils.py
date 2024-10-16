import torch

from models.glow_model.model import Glow
from models.pixelCNN_model.main import PixelCNN
from models.VAE_model.model import SimpleVAE
from models.diffusion_model.model import DiffusionModel


model_class_dict = {
    "glow": Glow,
    "PixelCNN": PixelCNN,
    "vae": SimpleVAE,
    "diffusion": DiffusionModel
}

model_classes = model_class_dict.keys()


def load_generative_model(model_type, model_name):
    model_class = model_class_dict[model_type]

    return model_class.load_serialised(model_name)


def compute_nll(dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=1)

    device = torch.device("cuda")
    print(f"using device: {device}")

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


def n_parameters(model):
    return sum(
        p.numel() for p in model.parameters()
    )
