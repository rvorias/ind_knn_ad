import numpy as np
import torch
from loguru import logger

from indad.data import MVTecDataset
from indad.models import EXPORT_DIR, SPADE
from indad.utils import run_onnx, run_torchscript


def test_spade():
    model = SPADE()

    train_ds, test_ds = MVTecDataset("hazelnut_reduced").get_dataloaders()

    model.fit(train_ds)

    results = model.evaluate(test_ds)

    logger.info(results)


def test_spade_export():
    model = SPADE()
    export_name = "spade_test_model"

    train_ds, test_ds = MVTecDataset("hazelnut_reduced").get_dataloaders()

    model.fit(train_ds)

    model.export(export_name)

    sample = torch.randn(1, 3, 224, 224)

    z_score_onnx, s_map_onnx = run_onnx(EXPORT_DIR / f"{export_name}.onnx", sample)
    z_score_torchscript, s_map_torchscript = run_torchscript(
        EXPORT_DIR / f"{export_name}.pt", sample
    )

    print(s_map_onnx)
    print(s_map_torchscript)

    mse_error = np.mean((s_map_onnx - s_map_torchscript.numpy()) ** 2)
    print(f"MSE error between s_map_onnx and s_map_torchscript: {mse_error}")
    largest_error = np.max(np.abs(s_map_onnx - s_map_torchscript.numpy()))
    print(f"Largest error between s_map_onnx and s_map_torchscript: {largest_error}")

    assert np.allclose(z_score_onnx, z_score_torchscript, atol=1e-3)
    assert np.allclose(s_map_onnx, s_map_torchscript, atol=5e-2)
