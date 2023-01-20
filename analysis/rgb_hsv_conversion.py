import torch
from analysis_utils import get_vanilla_dataset

from matplotlib import pyplot as plt


from torch.autograd.functional import jacobian

# function from https://www.geeksforgeeks.org/program-change-rgb-color-model-hsv-color-model/

def rgb_to_hsv(r, g, b):

    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    # r, g, b = r / 255.0, g / 255.0, b / 255.0

    # h, s, v = hue, saturation, value
    # rgb = torch.cat([r, g, b])
    cmax = max(r, g, b)    # maximum of r, g, b
    cmin = min(r, g, b)    # minimum of r, g, b
    # cmax = cmax.values
    # cmin = cmin.values
    diff = cmax - cmin       # diff of cmax and cmin.

    # r_mask = (cmax == r)
    # g_mask = (cmax == g)
    # b_mask = (cmax == b)

    # h = r_mask * ((60 * ((g - b) / diff) + 360) % 360)
    # h += g_mask * ((60 * ((b - r) / diff) + 120) % 360)
    # h += b_mask * ((60 * ((r - g) / diff) + 240) % 360)

    # # if cmax and cmax are equal then h = 0
    if cmax - cmin < 0.1:
        return r, g, b

    # if cmax equal r then compute h
    if cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360

    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360

    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    #  cmax equal zero
    # if cmax == 0:
    #     diff += 0.01
    s = (diff / cmax)

    # compute v
    v = cmax
    return h/360, s, v


''' Driver Code '''
# print(rgb_to_hsv(45, 215, 0))
# print(rgb_to_hsv(31, 52, 29))


dataset = get_vanilla_dataset("cifar")


for img_index in range(2):
    x, y = dataset[img_index]
    x += 0.5

    J_ten = torch.zeros((32, 32))

    for i in range(32):
        for j in range(32):
            r, g, b = x[:, i, j]
            rgb_inputs = tuple([r, g, b])

            # J = jacobian(rgb_to_hsv, rgb_inputs)
            # J = torch.tensor(J)
            h, s, v = rgb_to_hsv(r, g, b)
            J_ten[i, j] = v + 0.1 # abs(J.det())

    log_mult = torch.log(J_ten)
    log_mult = log_mult.sum()
    print(img_index, "Likelihood sum:", log_mult)
    # print("Likelihood multiplier:", torch.exp(log_mult))


for boi in [8, 13]:
    x, y = dataset[boi]
    x += 0.5
    x = x.permute(1, 2, 0)
    plt.imshow(x)
    plt.axis("off")
    plt.savefig(f"cifar_image_{boi}.png")
    plt.show()
