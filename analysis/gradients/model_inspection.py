from analysis.analysis_utils import load_generative_model

from pandas import DataFrame


def descriptive_layer_table(model):
    return DataFrame(
        ((name, p.shape, len(p.flatten())) for name, p in model.named_parameters()),
        columns=["name", "shape", "params"],
    )


if __name__ == "__main__":
    my_pixelCNN = load_generative_model("PixelCNN", "PixelCNN_FashionMNIST_checkpoint.pt", "./pixelCNN_model/")

    my_glow = load_generative_model("glow", "glow_checkpoint_18740.pt", "./glow_model/FashionMNIST_stable/",
                                    image_shape=(28, 28, 1))

    my_vae = load_generative_model("vae", "VAE_FashionMNIST_checkpoint.pt", "./VAE_model/", input_shape=(28, 28, 1))

    for my_model in [my_pixelCNN, my_glow, my_vae]:
        table = descriptive_layer_table(my_model)

        print("total params:", table["params"].sum())

        print(table)  # [table["params"] > 10**4])
        print()
        print()
        print()
