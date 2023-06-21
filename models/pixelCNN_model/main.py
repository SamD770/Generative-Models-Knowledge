import os
import time
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils

from data.datasets import get_MNIST, get_FashionMNIST

from path_definitions import PIXEL_CNN_ROOT
from os import path

backends.cudnn.benchmark = True

from generative_model import GenerativeModel


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, kH // 2 + 1 :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class PixelCNN(nn.Module, GenerativeModel):
    def __init__(self, fm=64):
        super().__init__()
        self.fm = fm
        self.net = nn.Sequential(
            MaskedConv2d("A", 1, fm, 7, 1, 3, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
            MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
            MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
            MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
            MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
            MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
            MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
            MaskedConv2d("B", fm, fm, 7, 1, 3, bias=False),
            nn.BatchNorm2d(fm),
            nn.ReLU(True),
            nn.Conv2d(fm, 256, 1),
        )

        self.net.cuda()

    def eval_nll(self, x):
        x = Variable(x.cuda())
        target = Variable((x.data[:, 0] * 255).long())
        output = self.net(x)
        nll = F.cross_entropy(output, target)

        return nll

    def generate_sample(self, batch_size):
        sample = torch.Tensor(batch_size, 1, 28, 28).cuda()
        sample.fill_(0)
        self.net.train(False)
        with torch.no_grad():
            for i in range(28):
                for j in range(28):
                    out = self.net(Variable(sample))
                    probs = F.softmax(out[:, :, i, j]).data
                    sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255
        return sample

    @staticmethod
    def load_serialised(model_name, **params):

        save_path = path.join(model_name, save_file)

        model = PixelCNN()
        state_dict = torch.load(save_path)
        model.net.load_state_dict(state_dict)

        return model


if __name__ == "__main__":
    fm = 64

    model = PixelCNN()

    _, _, train_dataset, test_dataset = get_MNIST("../")

    tr = data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=1, pin_memory=True
    )
    te = data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True
    )
    sample = torch.Tensor(144, 1, 28, 28).cuda()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(50):
        # train
        err_tr = []
        cuda.synchronize()
        time_tr = time.time()
        model.train(True)
        for input, _ in tr:
            loss = model.eval_nll(input)
            err_tr.append(loss.data.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        cuda.synchronize()
        time_tr = time.time() - time_tr

        # compute error on test set
        err_te = []
        cuda.synchronize()
        time_te = time.time()
        model.net.train(False)
        for input, _ in te:
            with torch.no_grad():
                loss = model.eval_nll(input)
                err_te.append(loss.data.item())
        cuda.synchronize()
        time_te = time.time() - time_te

        # sample
        sample = model.generate_sample(32).cpu()

        utils.save_image(sample, "sample_{:02d}.png".format(epoch), nrow=12, padding=0)
        print(
            f"epoch={epoch}      "
            f"nll_tr={np.mean(err_tr):.7f}       "
            f"nll_te={np.mean(err_te):.7f}       "
            f"time_tr={time_tr:.1f}s         "
            f"time_te={time_te:.1f}s"
        )

        torch.save(model.net.state_dict(), f"PixelCNN_MNIST_checkpoint.pt")
        torch.save(optimizer.state_dict(), f"optimizer_checkpoint.pt")
