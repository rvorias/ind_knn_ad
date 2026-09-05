from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from indad.utils import NativeGaussianBlur, get_coreset_idx_randomp, get_tqdm_params

EXPORT_DIR = Path("./exports")


class KNNExtractor(torch.nn.Module):
    """Shared feature-extraction and evaluation behavior for anomaly detectors."""

    _dynamic_buffers: tuple[str, ...] = ()

    def __init__(
        self,
        backbone_name: str = "resnet50",
        out_indices: tuple[int, ...] | None = None,
        pool_last: bool = False,
        image_size: int = 224,
        device: str | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.feature_extractor = timm.create_model(
            backbone_name,
            out_indices=out_indices,
            features_only=True,
            pretrained=pretrained,
            exportable=True,
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()

        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
        self.backbone_name = backbone_name
        self.out_indices = out_indices
        self.image_size = image_size
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = self.feature_extractor.to(self.device)
        self._is_fitted = False

    def train(self, mode: bool = True):
        """Keep the frozen pretrained backbone in evaluation mode."""
        super().train(mode)
        self.feature_extractor.eval()
        return self

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        reference = next(self.feature_extractor.parameters(), None)
        if reference is None:
            reference = next(self.feature_extractor.buffers(), None)
        if reference is not None:
            self.device = str(reference.device)
        return module

    def extract(self, x: Tensor):
        with torch.no_grad():
            feature_maps = self.feature_extractor(x.to(self.device))
        feature_maps = [feature_map.cpu() for feature_map in feature_maps]
        if self.pool is not None:
            return feature_maps[:-1], self.pool(feature_maps[-1])
        return feature_maps

    @staticmethod
    def _require_single_image(sample: Tensor) -> None:
        if sample.ndim != 4 or sample.shape[0] != 1:
            raise ValueError("Models currently require input shaped (1, C, H, W)")

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before running inference")

    def fit(self, _: DataLoader):
        raise NotImplementedError

    def predict(self, sample: Tensor):
        return self(sample)

    def evaluate(self, test_dl: DataLoader) -> tuple[float, float]:
        """Evaluate image- and pixel-level ROC AUC on a labelled dataset."""
        self._require_fitted()
        image_preds: list[float] = []
        image_labels: list[int] = []
        pixel_preds: list[np.ndarray] = []
        pixel_labels: list[np.ndarray] = []

        for sample, mask, label in tqdm(test_dl, **get_tqdm_params()):
            self._require_single_image(sample)
            image_score, score_map = self(sample)

            score_values = score_map.detach().cpu().reshape(-1).numpy()
            label_values = mask.detach().cpu().reshape(-1).numpy()
            if score_values.size != label_values.size:
                raise ValueError(
                    "Prediction and mask sizes differ: "
                    f"{score_values.size} != {label_values.size}"
                )

            image_preds.append(float(image_score.detach().cpu()))
            image_labels.extend(label.detach().cpu().reshape(-1).int().tolist())
            pixel_preds.append(score_values)
            pixel_labels.append(label_values)

        if len(set(image_labels)) < 2:
            raise ValueError("Image ROC AUC requires both normal and anomalous samples")

        flat_pixel_labels = np.concatenate(pixel_labels)
        if np.unique(flat_pixel_labels).size < 2:
            raise ValueError("Pixel ROC AUC requires both normal and anomalous pixels")

        image_rocauc = roc_auc_score(image_labels, image_preds)
        pixel_rocauc = roc_auc_score(flat_pixel_labels, np.concatenate(pixel_preds))
        return float(image_rocauc), float(pixel_rocauc)

    def get_parameters(self, extra_params: dict[str, Any] | None = None) -> dict:
        return {
            "backbone_name": self.backbone_name,
            "out_indices": self.out_indices,
            "image_size": self.image_size,
            "pretrained": self.pretrained,
            "device": self.device,
            **(extra_params or {}),
        }

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Learned memory banks have data-dependent shapes. Resize their registered
        # buffers before PyTorch performs its normal state-dict copy.
        for name in self._dynamic_buffers:
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._is_fitted = bool(
            self._dynamic_buffers and getattr(self, self._dynamic_buffers[0]).numel()
        )


class SPADE(KNNExtractor):
    _dynamic_buffers = ("z_lib", "feature_map_0", "feature_map_1", "feature_map_2")

    def __init__(
        self,
        k: int = 5,
        backbone_name: str = "resnet18",
        image_size: int = 224,
        device: str | None = None,
        pretrained: bool = True,
    ):
        if k <= 0:
            raise ValueError("k must be positive")
        super().__init__(
            backbone_name=backbone_name,
            out_indices=(1, 2, 3, -1),
            pool_last=True,
            image_size=image_size,
            device=device,
            pretrained=pretrained,
        )
        self.k = k
        self.blur = NativeGaussianBlur()
        self.register_buffer("z_lib", torch.empty(0))
        self.register_buffer("feature_map_0", torch.empty(0))
        self.register_buffer("feature_map_1", torch.empty(0))
        self.register_buffer("feature_map_2", torch.empty(0))

    def fit(self, train_dl: DataLoader):
        memory_device = self.z_lib.device
        z_parts: list[Tensor] = []
        feature_parts: list[list[Tensor]] = [[], [], []]

        for sample, *_ in tqdm(train_dl, **get_tqdm_params()):
            self._require_single_image(sample)
            feature_maps, z = self.extract(sample)
            if len(feature_maps) != len(feature_parts):
                raise ValueError("SPADE expects exactly three spatial feature maps")
            z_parts.append(z.flatten(1))
            for index, feature_map in enumerate(feature_maps):
                feature_parts[index].append(feature_map)

        if not z_parts:
            raise ValueError("Training data must contain at least one image")

        self.z_lib = torch.cat(z_parts, dim=0).to(memory_device)
        self.feature_map_0 = torch.cat(feature_parts[0], dim=0).to(memory_device)
        self.feature_map_1 = torch.cat(feature_parts[1], dim=0).to(memory_device)
        self.feature_map_2 = torch.cat(feature_parts[2], dim=0).to(memory_device)
        self._is_fitted = True
        return self

    def forward(self, sample: Tensor):
        self._require_fitted()
        self._require_single_image(sample)
        feature_maps, z = self.extract(sample)
        feature_maps = [
            feature_map.to(self.z_lib.device) for feature_map in feature_maps
        ]
        z = z.to(self.z_lib.device)

        distances = torch.linalg.vector_norm(self.z_lib - z.flatten(1)[0], dim=1)
        n_neighbors = min(self.k, self.z_lib.shape[0])
        values, indices = torch.topk(distances, n_neighbors, largest=False)
        image_score = values.mean()

        score_map = torch.zeros(
            (1, 1, sample.shape[-2], sample.shape[-1]),
            dtype=z.dtype,
            device=self.z_lib.device,
        )
        feature_libraries = [
            self.feature_map_0,
            self.feature_map_1,
            self.feature_map_2,
        ]
        for feature_map, feature_library in zip(feature_maps, feature_libraries):
            nearest_feature_maps = torch.index_select(feature_library, 0, indices)
            layer_map = torch.linalg.vector_norm(
                nearest_feature_maps - feature_map, dim=1
            ).amin(dim=0, keepdim=True)
            score_map += torch.nn.functional.interpolate(
                layer_map.unsqueeze(0),
                size=(sample.shape[-2], sample.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

        return image_score, self.blur(score_map)

    def get_parameters(self):
        return super().get_parameters({"k": self.k})

    def export(self, save_name: str, export_dir: Path = EXPORT_DIR) -> dict[str, Path]:
        self._require_fitted()
        export_dir.mkdir(parents=True, exist_ok=True)

        torchscript_path = export_dir / f"{save_name}.pt"
        torch.jit.script(self).save(str(torchscript_path))

        onnx_path = export_dir / f"{save_name}.onnx"
        tensor_x = torch.rand((1, 3, self.image_size, self.image_size))
        onnx_program = torch.onnx.dynamo_export(self, tensor_x)
        onnx_program.save(str(onnx_path))
        return {"torchscript": torchscript_path, "onnx": onnx_path}


class PaDiM(KNNExtractor):
    _dynamic_buffers = ("r_indices", "means", "covariance_inv", "feature_map_size")

    def __init__(
        self,
        d_reduced: int = 100,
        backbone_name: str = "resnet18",
        epsilon: float = 0.04,
        random_seed: int = 0,
        image_size: int = 224,
        device: str | None = None,
        pretrained: bool = True,
    ):
        if d_reduced <= 0:
            raise ValueError("d_reduced must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        super().__init__(
            backbone_name=backbone_name,
            out_indices=(1, 2, 3),
            image_size=image_size,
            device=device,
            pretrained=pretrained,
        )
        self.d_reduced = d_reduced
        self.epsilon = epsilon
        self.random_seed = random_seed
        self.register_buffer("r_indices", torch.empty(0, dtype=torch.long))
        self.register_buffer("means", torch.empty(0))
        self.register_buffer("covariance_inv", torch.empty(0))
        self.register_buffer("feature_map_size", torch.empty(0, dtype=torch.long))

    def fit(self, train_dl: DataLoader):
        memory_device = self.means.device
        patch_parts: list[Tensor] = []
        target_size: tuple[int, int] | None = None

        for sample, *_ in tqdm(train_dl, **get_tqdm_params()):
            self._require_single_image(sample)
            feature_maps = self.extract(sample)
            if target_size is None:
                target_size = tuple(feature_maps[0].shape[-2:])
            resized_maps = [
                torch.nn.functional.adaptive_avg_pool2d(feature_map, target_size)
                for feature_map in feature_maps
            ]
            patch_parts.append(torch.cat(resized_maps, dim=1))

        if len(patch_parts) < 2:
            raise ValueError("PaDiM requires at least two training images")
        assert target_size is not None

        patch_library = torch.cat(patch_parts, dim=0).to(memory_device)
        total_dimensions = patch_library.shape[1]
        effective_dimensions = min(self.d_reduced, total_dimensions)
        if effective_dimensions < total_dimensions:
            generator = torch.Generator().manual_seed(self.random_seed)
            self.r_indices = torch.randperm(
                total_dimensions, generator=generator, device="cpu"
            )[:effective_dimensions].to(memory_device)
        else:
            self.r_indices = torch.arange(total_dimensions, device=memory_device)

        patch_library = patch_library[:, self.r_indices]
        self.means = patch_library.mean(dim=0, keepdim=True)
        centered = patch_library - self.means
        covariance = torch.einsum("nchw,ndhw->cdhw", centered, centered) / (
            patch_library.shape[0] - 1
        )
        covariance += self.epsilon * torch.eye(
            effective_dimensions,
            dtype=covariance.dtype,
            device=covariance.device,
        ).unsqueeze(-1).unsqueeze(-1)
        self.covariance_inv = torch.linalg.inv(covariance.permute(2, 3, 0, 1)).permute(
            2, 3, 0, 1
        )
        self.feature_map_size = torch.tensor(
            target_size, dtype=torch.long, device=memory_device
        )
        self._is_fitted = True
        return self

    def forward(self, sample: Tensor):
        self._require_fitted()
        self._require_single_image(sample)
        feature_maps = self.extract(sample)
        feature_maps = [
            feature_map.to(self.means.device) for feature_map in feature_maps
        ]
        target_size = (
            int(self.feature_map_size[0]),
            int(self.feature_map_size[1]),
        )
        resized_maps = [
            torch.nn.functional.adaptive_avg_pool2d(feature_map, target_size)
            for feature_map in feature_maps
        ]
        centered = torch.cat(resized_maps, dim=1)[:, self.r_indices] - self.means

        left = torch.einsum("nchw,cdhw->ndhw", centered, self.covariance_inv)
        score_map = torch.sqrt(
            torch.einsum("nchw,nchw->nhw", left, centered).clamp_min(0)
        )
        scaled_score_map = torch.nn.functional.interpolate(
            score_map.unsqueeze(1),
            size=(sample.shape[-2], sample.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return score_map.amax(), scaled_score_map

    def get_parameters(self):
        parameters: dict[str, Any] = {
            "d_reduced": self.d_reduced,
            "epsilon": self.epsilon,
            "random_seed": self.random_seed,
        }
        if self.r_indices.numel():
            parameters["effective_dimensions"] = int(self.r_indices.numel())
        return super().get_parameters(parameters)


class PatchCore(KNNExtractor):
    _dynamic_buffers = ("patch_lib", "feature_map_size")

    def __init__(
        self,
        f_coreset: float = 0.01,
        backbone_name: str = "resnet18",
        coreset_eps: float = 0.90,
        n_reweight: int = 3,
        distance_chunk_size: int = 2048,
        random_seed: int = 0,
        image_size: int = 224,
        device: str | None = None,
        pretrained: bool = True,
    ):
        if not 0 < f_coreset <= 1:
            raise ValueError("f_coreset must be in the interval (0, 1]")
        if n_reweight <= 0:
            raise ValueError("n_reweight must be positive")
        if distance_chunk_size <= 0:
            raise ValueError("distance_chunk_size must be positive")
        super().__init__(
            backbone_name=backbone_name,
            out_indices=(2, 3),
            image_size=image_size,
            device=device,
            pretrained=pretrained,
        )
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps
        self.n_reweight = n_reweight
        self.distance_chunk_size = distance_chunk_size
        self.random_seed = random_seed
        self.average = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.blur = NativeGaussianBlur()
        self.register_buffer("patch_lib", torch.empty(0))
        self.register_buffer("feature_map_size", torch.empty(0, dtype=torch.long))

    def fit(self, train_dl: DataLoader):
        memory_device = self.patch_lib.device
        patch_parts: list[Tensor] = []
        target_size: tuple[int, int] | None = None

        for sample, *_ in tqdm(train_dl, **get_tqdm_params()):
            self._require_single_image(sample)
            feature_maps = self.extract(sample)
            averaged_maps = [self.average(feature_map) for feature_map in feature_maps]
            if target_size is None:
                target_size = tuple(averaged_maps[0].shape[-2:])
            resized_maps = [
                torch.nn.functional.adaptive_avg_pool2d(feature_map, target_size)
                for feature_map in averaged_maps
            ]
            patch = torch.cat(resized_maps, dim=1)
            patch_parts.append(patch.flatten(2).squeeze(0).T)

        if not patch_parts:
            raise ValueError("Training data must contain at least one image")
        assert target_size is not None

        self.patch_lib = torch.cat(patch_parts, dim=0).to(memory_device)
        if self.f_coreset < 1:
            n_coreset = max(1, int(self.f_coreset * self.patch_lib.shape[0]))
            coreset_idx = get_coreset_idx_randomp(
                self.patch_lib,
                n=n_coreset,
                eps=self.coreset_eps,
                random_state=self.random_seed,
            )
            self.patch_lib = self.patch_lib[coreset_idx.to(memory_device)]

        self.feature_map_size = torch.tensor(
            target_size, dtype=torch.long, device=memory_device
        )
        self._is_fitted = True
        return self

    def forward(self, sample: Tensor):
        self._require_fitted()
        self._require_single_image(sample)
        feature_maps = self.extract(sample)
        feature_maps = [
            feature_map.to(self.patch_lib.device) for feature_map in feature_maps
        ]
        target_size = (
            int(self.feature_map_size[0]),
            int(self.feature_map_size[1]),
        )
        resized_maps = [
            torch.nn.functional.adaptive_avg_pool2d(
                self.average(feature_map), target_size
            )
            for feature_map in feature_maps
        ]
        patch = torch.cat(resized_maps, dim=1).flatten(2).squeeze(0).T

        min_values = torch.full(
            (patch.shape[0],), float("inf"), dtype=patch.dtype, device=patch.device
        )
        min_indices = torch.zeros(patch.shape[0], dtype=torch.long, device=patch.device)
        for start in range(0, self.patch_lib.shape[0], self.distance_chunk_size):
            chunk = self.patch_lib[start : start + self.distance_chunk_size]
            chunk_values, chunk_indices = torch.min(torch.cdist(patch, chunk), dim=1)
            update = chunk_values < min_values
            min_values = torch.where(update, chunk_values, min_values)
            min_indices = torch.where(update, chunk_indices + start, min_indices)
        max_distance, max_index = torch.max(min_values, dim=0)

        test_patch = patch[max_index].unsqueeze(0)
        nearest_patch = self.patch_lib[min_indices[max_index]].unsqueeze(0)
        memory_distances = torch.cdist(nearest_patch, self.patch_lib)
        n_neighbors = min(self.n_reweight, self.patch_lib.shape[0])
        if n_neighbors == 1:
            image_score = max_distance
        else:
            _, neighbor_indices = torch.topk(
                memory_distances, k=n_neighbors, largest=False
            )
            neighbor_distances = torch.linalg.vector_norm(
                test_patch - self.patch_lib[neighbor_indices[0]], dim=1
            )
            scale = torch.sqrt(
                torch.tensor(
                    patch.shape[1],
                    dtype=neighbor_distances.dtype,
                    device=patch.device,
                )
            )
            nearest_weight = torch.softmax(neighbor_distances / scale, dim=0)[0]
            image_score = (1 - nearest_weight) * max_distance

        score_map = min_values.view(1, 1, *target_size)
        score_map = torch.nn.functional.interpolate(
            score_map,
            size=(sample.shape[-2], sample.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        return image_score, self.blur(score_map)

    def get_parameters(self):
        return super().get_parameters(
            {
                "f_coreset": self.f_coreset,
                "coreset_eps": self.coreset_eps,
                "n_reweight": self.n_reweight,
                "distance_chunk_size": self.distance_chunk_size,
                "random_seed": self.random_seed,
            }
        )

    def export(self, save_name: str, export_dir: Path = EXPORT_DIR) -> dict[str, Path]:
        """Export PatchCore to its currently supported TorchScript format."""
        self._require_fitted()
        export_dir.mkdir(parents=True, exist_ok=True)
        torchscript_path = export_dir / f"{save_name}.pt"
        torch.jit.script(self).save(str(torchscript_path))
        return {"torchscript": torchscript_path}
