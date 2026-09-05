import pytest
import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FakeBackbone(nn.Module):
    """Small deterministic TIMM stand-in that remains TorchScript compatible."""

    out_indices: list[int]

    def __init__(self, out_indices):
        super().__init__()
        self.out_indices = list(out_indices)

    @staticmethod
    def _feature(x: Tensor, channels: int, height: int, width: int) -> Tensor:
        base = F.adaptive_avg_pool2d(x.mean(dim=1, keepdim=True), (height, width))
        scales = torch.arange(1, channels + 1, dtype=x.dtype, device=x.device)
        return base * scales.view(1, channels, 1, 1)

    def forward(self, x: Tensor) -> list[Tensor]:
        all_features = [
            self._feature(x, 2, 16, 16),
            self._feature(x, 3, 12, 12),
            self._feature(x, 4, 8, 8),
            self._feature(x, 5, 4, 4),
            self._feature(x, 6, 2, 2),
        ]
        selected: list[Tensor] = []
        for index in self.out_indices:
            selected.append(all_features[index])
        return selected


@pytest.fixture(autouse=True)
def fake_timm_backbone(monkeypatch):
    # Constructors call timm.create_model, so no test downloads pretrained weights.
    monkeypatch.setattr(
        timm,
        "create_model",
        lambda *args, **kwargs: FakeBackbone(kwargs["out_indices"]),
    )
