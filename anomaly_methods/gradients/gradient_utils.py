from torch import Tensor

from generative_model import GenerativeModel


def backprop_nll(model: GenerativeModel, batch: Tensor):
    nll = model.eval_nll(batch)
    model.zero_grad()
    nll.sum().backward()
