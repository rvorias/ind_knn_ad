from collections.abc import Callable

import pytest
import torch

from indad.models import SPADE, PaDiM, PatchCore
from indad.utils import run_onnx, run_torchscript


def training_data(count: int = 3):
    generator = torch.Generator().manual_seed(7)
    return [
        (torch.rand((1, 3, 32, 40), generator=generator), torch.tensor([0]))
        for _ in range(count)
    ]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SPADE(k=10, image_size=32, device="cpu"),
        lambda: PaDiM(d_reduced=12, image_size=32, device="cpu"),
        lambda: PatchCore(
            f_coreset=1.0, distance_chunk_size=17, image_size=32, device="cpu"
        ),
    ],
    ids=["spade", "padim", "patchcore"],
)
def test_fit_predict_and_state_dict_round_trip(factory: Callable):
    model = factory().fit(training_data())
    sample = torch.rand((1, 3, 28, 36), generator=torch.Generator().manual_seed(9))

    score, score_map = model.predict(sample)
    assert score.ndim == 0
    assert score_map.shape == (1, 1, 28, 36)
    assert torch.isfinite(score)
    assert torch.isfinite(score_map).all()
    assert score >= 0

    restored = factory()
    restored.load_state_dict(model.state_dict())
    restored_score, restored_map = restored(sample)
    torch.testing.assert_close(restored_score, score)
    torch.testing.assert_close(restored_map, score_map)


@pytest.mark.parametrize("d_reduced", [12, 100])
def test_padim_supports_equal_or_larger_requested_dimension(d_reduced: int):
    model = PaDiM(d_reduced=d_reduced, device="cpu").fit(training_data())
    score, score_map = model(torch.zeros(1, 3, 32, 40))

    assert model.r_indices.numel() == 12
    assert score_map.shape == (1, 1, 32, 40)
    assert torch.isfinite(score)


def test_spade_clamps_neighbors_to_small_training_gallery():
    model = SPADE(k=50, device="cpu").fit(training_data(count=2))
    score, _ = model(torch.zeros(1, 3, 32, 40))
    assert torch.isfinite(score)


def test_patchcore_single_neighbor_keeps_anomaly_distance():
    model = PatchCore(f_coreset=1.0, n_reweight=1, device="cpu").fit(
        [(torch.zeros(1, 3, 32, 40), torch.tensor([0]))]
    )
    score, _ = model(torch.ones(1, 3, 32, 40))
    assert score > 0


def test_models_reject_batches_and_inference_before_fit():
    model = PatchCore(f_coreset=1.0, device="cpu")
    with pytest.raises(RuntimeError, match="fit"):
        model(torch.zeros(1, 3, 32, 40))

    model.fit(training_data())
    with pytest.raises(ValueError, match=r"\(1, C, H, W\)"):
        model(torch.zeros(2, 3, 32, 40))


def test_refit_replaces_previous_memory_bank():
    model = SPADE(device="cpu").fit(training_data(count=2))
    assert model.z_lib.shape[0] == 2
    model.fit(training_data(count=3))
    assert model.z_lib.shape[0] == 3


def test_evaluate_returns_bounded_metrics():
    model = SPADE(device="cpu").fit(
        [(torch.zeros(1, 3, 32, 40), torch.tensor([0])) for _ in range(2)]
    )
    test_data = [
        (
            torch.zeros(1, 3, 32, 40),
            torch.zeros(1, 1, 32, 40),
            torch.tensor([0]),
        ),
        (
            torch.ones(1, 3, 32, 40),
            torch.ones(1, 1, 32, 40),
            torch.tensor([1]),
        ),
    ]

    image_auc, pixel_auc = model.evaluate(test_data)
    assert 0 <= image_auc <= 1
    assert 0 <= pixel_auc <= 1


def test_patchcore_exports_only_supported_torchscript_format(tmp_path):
    model = PatchCore(
        f_coreset=1.0, distance_chunk_size=17, image_size=32, device="cpu"
    ).fit(training_data())
    paths = model.export("patchcore", tmp_path)

    assert set(paths) == {"torchscript"}
    assert paths["torchscript"].is_file()
    score, score_map = run_torchscript(paths["torchscript"], torch.zeros(1, 3, 32, 32))
    assert torch.isfinite(score)
    assert score_map.shape == (1, 1, 32, 32)


@pytest.mark.filterwarnings("ignore:Node .* does not reference an nn.Module.*")
@pytest.mark.filterwarnings("ignore:Attempted to insert a get_attr Node.*")
@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:torch.onnx.dynamo_export only implements.*")
@pytest.mark.export
def test_spade_onnx_matches_torchscript(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    model = SPADE(image_size=32, device="cpu").fit(training_data())
    paths = model.export("spade", tmp_path)
    sample = torch.zeros(1, 3, 32, 32)

    onnx_score, onnx_map = run_onnx(paths["onnx"], sample)
    torch_score, torch_map = run_torchscript(paths["torchscript"], sample)
    assert torch.allclose(torch.as_tensor(onnx_score), torch_score, atol=1e-4)
    assert torch.allclose(torch.as_tensor(onnx_map), torch_map, atol=1e-4)
