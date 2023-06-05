from anomaly_methods import *

import matplotlib.pyplot as plt

model, hparams = load_glow_model(
    "../glow_model/svhn_glow/", "glow_checkpoint_286000.pt"
)


plt.figure(figsize=(20, 10))
plt.title("Histogram Glow - trained on SVHN")
plt.xlabel("Negative bits per dimension")

for dataset, name in zip([vanilla_test_svhn, vanilla_test_cifar], ["svhn", "cifar"]):
    nll = compute_nll(dataset, model, hparams)
    print(f"{name} NLL:", torch.mean(nll))
    plt.hist(-nll.numpy(), label=name, density=True, alpha=0.6, bins=30)

plt.legend()
plt.savefig("plots/glow_trained_svhn.png", dpi=300)
