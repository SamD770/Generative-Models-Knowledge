from .gradient_serialisation import (
    LayerFIMStore,
    serialise_gradients,
    get_save_file_name,
)
from ..analysis_utils import SampleDataset, load_generative_model


if __name__ == "__main__":
    model = load_generative_model("PixelCNN", "PixelCNN_FashionMNIST_checkpoint.pt", "pixelCNN_model/")

    for layer_name in [
        "net.22.weight",
        "net.22.bias",
        "net.16.weight",
        "net.16.bias",
        "net.10.weight",
        "net.10.bias",
        "net.4.weight",
        "net.4.bias",
    ]:
        batch_size = 1

        sample_dataset = SampleDataset(model, batch_count=32)
        FIM_store = LayerFIMStore(model, layer_name)

        save_file = get_save_file_name(
            model_name="PixelCNN_FashionMNIST",
            dataset_name=layer_name,
            batch_size=batch_size,
            method="FIM(un-normed)",
            filetype="png",
        )

        serialise_gradients(
            model,
            sample_dataset,
            save_file,
            FIM_store,
            batch_size,
            save_dir="./anomaly_methods/plots/FIM_plots/",
        )
