from analysis import *


import matplotlib.pyplot as plt


# svhn_model, svhn_hparams = load_glow_model("svhn_glow/", "glow_checkpoint_284856.pt")
cifar_model, cifar_hparams = load_glow_model("cifar_glow/", "glow_checkpoint_195250.pt")


cifar_samples = SampleDataset(cifar_model, batch_count=512)

# cifar_images = postprocess(cifar_model(temperature=1, reverse=True)).cpu()


plt.figure(figsize=(20, 10))
plt.title("Histogram Glow - trained on CIFAR")
plt.xlabel("Negative bits per dimension")


for dataset, name in zip(
        [vanilla_test_cifar, cifar_samples, vanilla_test_svhn],
        ["cifar data", "cifar samples", "svhn data"]):
    nll = compute_nll(dataset, cifar_model, cifar_hparams)
    print(nll[:10])
    print(f"{name} NLL:", torch.mean(nll))
    plt.hist(-nll.numpy(), label=name, density=True, alpha=0.6, bins=30)


plt.legend()

plt.savefig("plots/cifar_samples_vs_data.png", dpi=300)
