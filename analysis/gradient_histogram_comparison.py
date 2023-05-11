import matplotlib.pyplot as plt


from .gradients.unsupervised_sklearn import *


model_names = ["cifar_long", "svhn_working", "imagenet32", "celeba"]


dataset_names = ["cifar", "svhn", "imagenet32", "celeba"]


fig, axs = plt.subplots(nrows=4, ncols=1)

top_likelihood_ax = axs[0]  # , top_sample_ax = axs[0]
last_score_ax = axs[-1]

batch_size = 1


for model_name, id_dataset, ax in zip(model_names, dataset_names, axs):
    score_ax = ax  # , sample_ax = ax

    score_ax.sharex(top_likelihood_ax)

    score_ax.set_xticklabels([])
    score_ax.set_yticks([])

    score_ax.set_ylabel(f"{model_name}")

    id_test_data, id_train_data, ood_data_list = load_sklearn_norms(
        batch_size, model_name, id_dataset, dataset_names, printout=True
    )

    sklearn_model = OneClassSVM(nu=0.001)

    sklearn_model.fit(id_train_data)

    id_test_scores = sklearn_model.score_samples(id_test_data)

    score_ax.hist(id_test_scores)


last_score_ax.set_xlabel("gradient anomaly score value")

fig.legend(title="evaluation dataset")
# fig.tight_layout()

plt.savefig("./analysis/plots/gradient_hisogram_comparison.png")

print("done")
