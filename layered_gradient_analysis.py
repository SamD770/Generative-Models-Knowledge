import torch
import matplotlib.pyplot as plt


def norm_dict(grad_dict):
    return {
        k: [
            (grad**2).sum() for grad in grad_dict[k]
        ]
        for k in grad_dict.keys()
    }


cifar_norms = torch.load("cifar_norms.pt")
svhn_norms = torch.load("svhn_norms.pt")


# fig, ax = plt.subplots()
#
# plt.hist(cifar_norms["flow.layers.1.actnorm.bias"],
#          label="in distribution(cifar)", density=True, alpha=0.6, bins=20)
# plt.hist(svhn_norms["flow.layers.1.actnorm.bias"],
#          label="out of distribution (svhn)", density=True, alpha=0.6, bins=20)
#
#
# ax.hist()