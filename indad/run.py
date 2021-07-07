import click
import torch
from typing import List

from data import MVTecDataset, mvtec_classes
from model import SPADE, PaDiM, PatchCore
from utils import serialize_results, write_results

import warnings # for some torch warnings regarding depreciation
warnings.filterwarnings("ignore")

ALL_CLASSES = mvtec_classes()
ALLOWED_METHODS = ["spade", "padim", "patchcore"]

def run_model(classes, method):
    results = {}
    for cls  in classes:
        if method == "spade":
            model = SPADE(
                k=50,
                backbone="wide_resnet101_2",
            )
        elif method == "padim":
            model = PaDiM(
                d_reduced=250,
                backbone="wide_resnet101_2",
            )
        elif method == "patchcore":
            model = PatchCore(
                f_coreset=.01, 
                backbone="wide_resnet101_2",
            )
        print(f"Running tests for {cls}.")
        train_ds, test_ds = MVTecDataset(cls).load()

        model.fit(train_ds)
        image_rocauc, pixel_rocauc = model.evaluate(test_ds)
        
        print(f"Test results {cls} - image_rocauc: {image_rocauc}, pixel_rocauc: {pixel_rocauc}")
        results[cls] = [float(image_rocauc), float(pixel_rocauc)]

    image_results = [v[0] for _, v in results.items()]
    average_image_roc_auc = sum(image_results)/len(image_results)
    image_results = [v[1] for _, v in results.items()]
    average_pixel_roc_auc = sum(image_results)/len(image_results)

    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
    }
    return total_results

@click.command()
@click.argument("method")
@click.option("--dataset", default="all", help="Dataset, defaults to all datasets.")
def cli_interface(method, dataset): 
    if dataset == "all":
        dataset = ALL_CLASSES
    else:
        assert dataset in ALL_CLASSES, "Dataset does not exist."
        dataset = [dataset]

    method = method.lower()
    assert method in ALLOWED_METHODS, f"Select from {ALLOWED_METHODS}."

    total_results = run_model(dataset, method)

    write_results(total_results, method)
    
if __name__ == "__main__":
    cli_interface()