from pixelCNN_model.main import PixelCNN

import torch.nn.functional as F

from torch.utils.data import DataLoader

from analysis_utils import get_vanilla_dataset

file = "PixelCNN_checkpoint.pt"

model_dir = "../pixelCNN_model/"

model = PixelCNN.load_serialised(file, model_dir)


dl = DataLoader(get_vanilla_dataset("FashionMNIST"), batch_size=5)

for x, _ in dl:
    nll = model.eval_nll(x)
    # print(target.shape, x.shape, output.shape)
    # print(target)
    # print(x)
    print(nll)

    break
