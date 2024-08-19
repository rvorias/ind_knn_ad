import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from loguru import logger
from onnxruntime import InferenceSession
from PIL import ImageFilter
from sklearn import random_projection
from torch import nn, tensor
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

TQDM_PARAMS = {
    "file": sys.stdout,
    "bar_format": "   {l_bar}{bar:10}{r_bar}{bar:-10b}",
}


def get_tqdm_params():
    return TQDM_PARAMS


class GaussianBlur:
    def __init__(self, radius: int = 4):
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = (
            self.load(self.unload(img[0] / map_max).filter(self.blur_kernel)) * map_max
        )
        return final_map


class NativeGaussianBlur(nn.Module):
    def __init__(self, channels: int = 1, kernel_size: int = 21, sigma: float = 4.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.register_buffer("kernel", self.create_gaussian_kernel())

    def create_gaussian_kernel(self):
        coords = torch.arange(self.kernel_size).float() - self.kernel_size // 2
        x = coords.repeat(self.kernel_size, 1)
        y = x.t()
        gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * self.sigma**2))
        kernel = gaussian / gaussian.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(
            self.channels, 1, 1, 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=self.padding, groups=self.channels)


def get_coreset_idx_randomp(
    z_lib: tensor,
    n: int = 1000,
    eps: float = 0.90,
    float16: bool = True,
    force_cpu: bool = False,
) -> tensor:
    """Returns n coreset idx for given z_lib.

    Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
    CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)

    Args:
        z_lib:      (n, d) tensor of patches.
        n:          Number of patches to select.
        eps:        Agression of the sparse random projection.
        float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
        force_cpu:  Force cpu, useful in case of GPU OOM.

    Returns:
        coreset indices
    """

    print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
    try:
        transformer = random_projection.SparseRandomProjection(eps=eps)
        z_lib = torch.tensor(transformer.fit_transform(z_lib))
        print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
    except ValueError:
        print("   Error: could not project vectors. Please increase `eps`.")

    select_idx = 0
    last_item = z_lib[select_idx : select_idx + 1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
    # The line below is not faster than linalg.norm, although i'm keeping it in for
    # future reference.
    # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

    if float16:
        last_item = last_item.half()
        z_lib = z_lib.half()
        min_distances = min_distances.half()
    if torch.cuda.is_available() and not force_cpu:
        last_item = last_item.to("cuda")
        z_lib = z_lib.to("cuda")
        min_distances = min_distances.to("cuda")

    for _ in tqdm(range(n - 1), **TQDM_PARAMS):
        distances = torch.linalg.norm(
            z_lib - last_item, dim=1, keepdims=True
        )  # broadcasting step
        # distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True) # broadcasting step
        min_distances = torch.minimum(distances, min_distances)  # iterative step
        select_idx = torch.argmax(min_distances)  # selection step

        # bookkeeping
        last_item = z_lib[select_idx : select_idx + 1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx.to("cpu"))

    return torch.stack(coreset_idx)


def print_and_export_results(results: dict, method: str):
    """Writes results to .yaml and serialized results to .txt."""

    print("\n   ╭────────────────────────────╮")
    print("   │      Results summary       │")
    print("   ┢━━━━━━━━━━━━━━━━━━━━━━━━━━━━┪")
    print(f"   ┃ average image rocauc: {results['average image rocauc']:.2f} ┃")
    print(f"   ┃ average pixel rocauc: {results['average pixel rocauc']:.2f} ┃")
    print("   ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")

    # write
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    name = f"{method}_{timestamp}"

    results_yaml_path = f"./results/{name}.yml"
    scoreboard_path = f"./results/{name}.txt"

    with open(results_yaml_path, "w") as outfile:
        yaml.safe_dump(results, outfile, default_flow_style=False)
    with open(scoreboard_path, "w") as outfile:
        outfile.write(serialize_results(results["per_class_results"]))

    print(f"   Results written to {results_yaml_path}")


def serialize_results(results: dict) -> str:
    """Serialize a results dict into something usable in markdown."""
    n_first_col = 20
    ans = []
    for k, v in results.items():
        s = k + " " * (n_first_col - len(k))
        s = s + f"| {v[0]*100:.1f}  | {v[1]*100:.1f}  |"
        ans.append(s)
    return "\n".join(ans)


def run_onnx(model_path: str | Path, sample: torch.Tensor):
    logger.info(f"Running ONNX model from {model_path}")

    sess = InferenceSession(model_path)
    sample = sample.numpy()
    z_score, s_map = sess.run(None, {"l_sample_": sample})
    logger.info(
        f"Prediction result - z_score: {z_score.item()}, s_map shape: {s_map.shape}"
    )

    return z_score, s_map


def run_torchscript(model_path: str | Path, sample: torch.Tensor):
    logger.info(f"Running torchscript model from {model_path}")

    loaded_predictor = torch.jit.load(model_path)
    z_score, s_map = loaded_predictor(sample)
    logger.info(
        f"Prediction result - z_score: {z_score.item()}, s_map shape: {s_map.shape}"
    )

    return z_score, s_map
