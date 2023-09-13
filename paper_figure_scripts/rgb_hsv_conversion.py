import torch
from torch import Tensor
from data.utils import get_test_dataset

from matplotlib import pyplot as plt

from torch.autograd.functional import jacobian


# function from https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/


def rgb_to_hsv(rgb_tensor: Tensor) -> Tensor:
    c_min = rgb_tensor.min()
    c_max = rgb_tensor.max()

    diff = c_max - c_min

    r, g, b = rgb_tensor
    max_channel = rgb_tensor.argmax().item()

    if max_channel == 0:      # r is the max value
        h = (g - b)/(diff * 6)
    elif max_channel == 1:    # g is the max value
        h = (b - r)/(diff * 6) + 1/3
    else:           # g is the max value
        h = (r - g)/(diff * 6) + 2/3

    torch.remainder(h, 1)

    s = diff/c_max
    v = c_max

    return torch.stack([h, s, v])


def rgb_to_hsv_test():

    r = 132.
    g = 153.
    b = 122.
    rgb_tensor = torch.tensor([r, g, b], dtype=torch.double)/255
    hsv_tensor = rgb_to_hsv(rgb_tensor)

    h, s, v = hsv_tensor

    h = (h * 360).item()
    s = (s * 100).item()
    v = (v * 100).item()

    print(h, s, v)

    assert abs(h - 100.) < 1
    assert abs(s - 20.) < 1
    assert abs(v - 60.) < 1


def rgb_to_hsv_volume_change(rgb_tensor):
    J = jacobian(rgb_to_hsv, rgb_tensor)
    return J.det().abs()


def to_rgb_tensor(r, g, b):
    return torch.tensor([r, g, b], dtype=torch.double, requires_grad=True) / 255


def image_rgb_to_hsv_volume_change(img):
    """Img should be an image with rgb colour channels scaled in the range [0, 1], returns the
    change in Bits Per Dimension (BPD) by transitioning to an HSV colour channel."""

    log_det_sum = 0

    for i in range(32):
        for j in range(32):
            rgb_tensor = img[:, i, j]

            # To make comparison fair, we add a small amount of noise to account for quantization.
            # Without this the computation of the hue blows up when r = g = b.
            rgb_tensor_dequantized = rgb_tensor + torch.randn((3,))/255
            det = rgb_to_hsv_volume_change(rgb_tensor_dequantized)

            # det = det.nan_to_num().clamp(-10**6, 10**6)

            log_det = det.log()
            log_det = log_det
            log_det_sum += log_det

    return log_det_sum.item() / (3 * 32 * 32)   # We want to compute in bits/dim


if __name__ == "__main__":

    fig, axs = plt.subplots(ncols=2)

    dataset = get_test_dataset("cifar10")
    imgs = [
        dataset[i][0] for i in range(2)
    ]

    for ax, img in zip(axs, imgs):
        img += 0.5

        delta_BPD = image_rgb_to_hsv_volume_change(img)

        ax.imshow(img.permute(1, 2, 0))

        ax.set_xticks([])
        ax.set_yticks([])
        x_label = "$\\Delta_{BPD} = " \
                  "\\frac{\\log p_{RGB}(\\mathbf{x}) - \\log p_{HSV}(\\mathbf{x})} {3 \\times 32 \\times 32}$ = " + \
                  f"{delta_BPD:.2f} "

        ax.set_xlabel(x_label, size=15)

    plt.show()
