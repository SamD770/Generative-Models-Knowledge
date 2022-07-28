import torch

from analysis import load_glow_model, vanilla_test_cifar, vanilla_test_svhn
from random import randint

from math import log

import matplotlib.pyplot as plt

cifar_model, cifar_hparams = load_glow_model("glow_model/cifar_glow/", "glow_checkpoint_195250.pt")

model = cifar_model


def backprop_nll(img):
    img = img.unsqueeze(0)
    _, nll, _ = model(img)
    model.zero_grad()
    nll.backward()



def get_grads(img):
    """Gets the gradient vector wrt one sample."""
    backprop_nll(img)
    return [
        p.grad for p in model.parameters() if p.requires_grad
    ]


def grad_dot_prod(delta_x, delta_y):
    """Returns the dot product of two gradient vectors."""
    return sum(
        (grad_x * grad_y).sum() for grad_x, grad_y in zip(delta_x, delta_y)
    )


#
# id_grads = []
# ood_grads = []

# for _ in range(1):
#     i = randint(0, 10000)
#     for dataset, grad_list in zip([vanilla_test_cifar(), vanilla_test_svhn()],
#                                   [id_grads, ood_grads]):
#         shape_dict = {}
#         img, _ = dataset[i]
#         delta = get_grads(img)
#         grad_list.append(log(grad_dot_prod(delta, delta)))
#
#         print(f"layer count: {len(delta)}")
#         for grad in delta:
#             if grad.shape in shape_dict:
#                 shape_dict[grad.shape] += 1
#             else:
#                 shape_dict[grad.shape] = 1
#         print(f"layer shapes: {shape_dict}")
#         parameter_count = sum(
#             torch.numel(grad) for grad in delta
#         )
#         print(f"parameter count {parameter_count}")
#         print(f"delta: {delta[:3]}")
#
#
# flownet, top_later = model.children()
#
# print(type(model), type(flownet), type(top_later))
#
# for child in flownet.children():
#     print(type(child.children()))

#
# plt.figure(figsize=(20, 10))
# plt.title("Histogram Glow - gradients in and out of distribution")
# plt.xlabel("gradient vector log $L^2$")
#
#
# plt.hist(id_grads, label="id", density=True, alpha=0.6, bins=50)
# plt.hist(ood_grads, label="od", density=True, alpha=0.6, bins=50)
# plt.legend()
#
# plt.show()
# plt.savefig("plots/gradients_proof_of_concept_2.png", dpi=300)