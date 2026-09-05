"""Industrial datasets and three anomaly baselines, loaded on demand."""

from importlib import import_module

from indad._version import __version__

__all__ = [
    "MVTecDataset",
    "PaDiM",
    "PatchCore",
    "SPADE",
    "StreamingDataset",
    "__version__",
]


def __getattr__(name):
    # Dataset tooling should not import Torch/TIMM or initialize model machinery.
    if name in {"MVTecDataset", "StreamingDataset"}:
        module = "indad.data"
    elif name in {"SPADE", "PaDiM", "PatchCore"}:
        module = "indad.models"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
