from models.utils import load_generative_model

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

temp = 1


model_name_list = ["VAE_cifar"]
file_list = ["VAE_cifar.pt"]

for model_name, file in zip(model_name_list, file_list):
    print("sampling from", model_name)
    model_dir = f"../models/VAE_model/{model_name}/"

    # model_dir = "../pixelCNN_model/"

    # model = PixelCNN.load_serialised(model_dir, file)

    # model, hparams = load_glow_model(model_dir, file, image_shape=(28, 28, 1))

    model = load_generative_model("vae", f"/{file}", "./VAE_model", input_shape=(32, 32, 3), latent_dims=64)

    model.to("cuda")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("params:", pytorch_total_params)

    samples = model.generate_sample(32).cpu()
    print(f"range {(samples.min(), samples.max())}")

    # samples = postprocess(model(temperature=1, reverse=True)).cpu()

    title = f"samples from {model_name} model"
    grid = make_grid(samples, nrow=8).permute(1, 2, 0) + 0.5

    # plt.title(title)
    plt.imshow(grid)
    plt.axis("off")
    plt.savefig("anomaly_methods/plots/sample_plots/" + title + ".png")

print("done")
