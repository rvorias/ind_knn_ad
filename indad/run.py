import random
from typing import List

import click
import numpy as np

# seeds
import torch

from indad.data import MVTEC_CLASSES, MVTecDataset
from indad.models import SPADE, PaDiM, PatchCore
from indad.utils import print_and_export_results

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import warnings  # for some torch warnings regarding depreciation

warnings.filterwarnings("ignore")

ALL_CLASSES = MVTEC_CLASSES.keys()
ALLOWED_METHODS = ["spade", "padim", "patchcore"]


def run_model(method: str, classes: List[str], backbone: str):
    results = {}

    for class_name in classes:
        if method == "spade":
            model = SPADE(
                k=50,
                backbone_name=backbone,
            )
        elif method == "padim":
            model = PaDiM(
                d_reduced=350,
                backbone_name=backbone,
            )
        elif method == "patchcore":
            model = PatchCore(
                f_coreset=0.10,
                backbone_name=backbone,
            )

        print(f"\n█│ Running {method} on {class_name} dataset.")
        print(f" ╰{'─'*(len(method)+len(class_name)+23)}\n")
        train_ds, test_ds = MVTecDataset(class_name).get_dataloaders()

        print("   Training ...")
        model.fit(train_ds)
        print("   Testing ...")
        image_rocauc, pixel_rocauc = model.evaluate(test_ds)

        print(f"\n   ╭{'─'*(len(class_name)+15)}┬{'─'*20}┬{'─'*20}╮")
        print(
            f"   │ Test results {class_name} │ image_rocauc: {image_rocauc:.2f} │ pixel_rocauc: {pixel_rocauc:.2f} │"
        )
        print(f"   ╰{'─'*(len(class_name)+15)}┴{'─'*20}┴{'─'*20}╯")
        results[class_name] = [float(image_rocauc), float(pixel_rocauc)]

    image_results = [v[0] for _, v in results.items()]
    average_image_roc_auc = sum(image_results) / len(image_results)
    image_results = [v[1] for _, v in results.items()]
    average_pixel_roc_auc = sum(image_results) / len(image_results)

    total_results = {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
    }
    return total_results


@click.command()
@click.argument("method")
@click.option(
    "--dataset", default="all", help="Dataset name, defaults to all datasets."
)
@click.option(
    "--backbone", default="wide_resnet50_2", help="The TIMM compatible backbone."
)
def cli_interface(method: str, dataset: str, backbone: str):
    if dataset == "all":
        dataset = ALL_CLASSES
    else:
        dataset = [dataset]

    method = method.lower()
    assert method in ALLOWED_METHODS, f"Select from {ALLOWED_METHODS}."

    total_results = run_model(method, dataset, backbone)

    print_and_export_results(total_results, method)


if __name__ == "__main__":
    cli_interface()
