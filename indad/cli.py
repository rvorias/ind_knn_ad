import random
from collections.abc import Iterable

import click
import numpy as np
import torch

from indad._version import __version__
from indad.data import MVTEC_CLASSES, MVTecDataset
from indad.models import SPADE, PaDiM, PatchCore
from indad.utils import print_and_export_results

ALL_CLASSES = tuple(MVTEC_CLASSES)
ALLOWED_METHODS = ("spade", "padim", "patchcore")


def _build_model(
    method: str,
    backbone: str,
    image_size: int,
    device: str | None,
    seed: int,
):
    common = {
        "backbone_name": backbone,
        "image_size": image_size,
        "device": device,
    }
    if method == "spade":
        return SPADE(k=50, **common)
    if method == "padim":
        return PaDiM(d_reduced=350, random_seed=seed, **common)
    if method == "patchcore":
        return PatchCore(f_coreset=0.10, random_seed=seed, **common)
    raise ValueError(f"Unknown method: {method}")


def run_model(
    method: str,
    classes: Iterable[str],
    backbone: str,
    image_size: int = 224,
    device: str | None = None,
    seed: int = 0,
):
    classes = tuple(classes)
    if not classes:
        raise ValueError("At least one dataset is required")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    results = {}
    model = None

    for class_name in classes:
        model = _build_model(method, backbone, image_size, device, seed)

        print(f"\n█│ Running {method} on {class_name} dataset.")
        print(f" ╰{'─' * (len(method) + len(class_name) + 23)}\n")
        train_dl, test_dl = MVTecDataset(class_name, size=image_size).get_dataloaders()

        print("   Training ...")
        model.fit(train_dl)
        print("   Testing ...")
        image_rocauc, pixel_rocauc = model.evaluate(test_dl)

        print(
            f"\n   {class_name}: image ROC AUC={image_rocauc:.3f}, pixel ROC AUC={pixel_rocauc:.3f}"
        )
        results[class_name] = [image_rocauc, pixel_rocauc]

    average_image_roc_auc = sum(value[0] for value in results.values()) / len(results)
    average_pixel_roc_auc = sum(value[1] for value in results.values()) / len(results)
    assert model is not None
    return {
        "per_class_results": results,
        "average image rocauc": average_image_roc_auc,
        "average pixel rocauc": average_pixel_roc_auc,
        "model parameters": model.get_parameters(),
        "seed": seed,
    }


@click.command()
@click.version_option(version=__version__, prog_name="indad")
@click.argument("method", type=click.Choice(ALLOWED_METHODS, case_sensitive=False))
@click.option(
    "--dataset", default="all", show_default=True, help="Dataset name or 'all'."
)
@click.option(
    "--backbone",
    default="wide_resnet50_2",
    show_default=True,
    help="TIMM-compatible feature backbone.",
)
@click.option(
    "--image-size", default=224, show_default=True, type=click.IntRange(min=1)
)
@click.option(
    "--device", default=None, help="Torch device, for example 'cpu' or 'cuda'."
)
@click.option("--seed", default=0, show_default=True, type=int)
def cli_interface(
    method: str,
    dataset: str,
    backbone: str,
    image_size: int,
    device: str | None,
    seed: int,
):
    """Train and evaluate an anomaly-detection METHOD."""
    classes = ALL_CLASSES if dataset == "all" else (dataset,)
    total_results = run_model(
        method.lower(),
        classes,
        backbone,
        image_size=image_size,
        device=device,
        seed=seed,
    )
    print_and_export_results(total_results, method.lower())
