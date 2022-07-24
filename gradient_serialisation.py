from gradients_analysis import *


gradient_count = 2000


# TODO: add likelihoods and full gradient vectors

for dataset, save_file in zip([vanilla_test_cifar(), vanilla_test_svhn()],
                              ["cifar_norms.pt", "svhn_norms.pt"]):

    print(f"creating {save_file}:")
    grad_dict = {
        name: [] for name, _ in model.named_parameters()
    }

    for i in range(gradient_count):
        img, _ = dataset[i]
        backprop_nll(img)

        for name, p in model.named_parameters():
            grad_dict[name].append(
                (p.grad**2).sum()
            )

        if i % (gradient_count//50) == 0:
            print(f"{i}/{gradient_count} complete")

    for key, value in grad_dict.items():
        grad_dict[key] = torch.tensor(value)

    torch.save(grad_dict, save_file)
    print("done")

