import pytest
import torch

from indad.utils import get_coreset_idx_randomp


def test_coreset_is_deterministic_and_unique():
    features = torch.rand((20, 100), generator=torch.Generator().manual_seed(4))
    first = get_coreset_idx_randomp(features, n=5, random_state=12, force_cpu=True)
    second = get_coreset_idx_randomp(features, n=5, random_state=12, force_cpu=True)

    torch.testing.assert_close(first, second)
    assert first.unique().numel() == 5


@pytest.mark.parametrize("n", [0, 21])
def test_coreset_validates_size(n: int):
    with pytest.raises(ValueError, match="n must be between"):
        get_coreset_idx_randomp(torch.zeros(20, 2), n=n)
