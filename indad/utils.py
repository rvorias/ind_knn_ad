import yaml
from tqdm import tqdm
from datetime import datetime

import torch
from torchvision import transforms

import numpy as np
from PIL import ImageFilter
from sklearn import random_projection


class GaussianBlur:
    def __init__(self, radius : int = 4):
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(
            self.unload(img[0]/map_max).filter(self.blur_kernel)
        )*map_max
        return final_map


def get_coreset_idx_randomp(z_lib, n : int = 3, eps : float = .95):
    """Returns n coreset idx for given z_lib.
    
    Might take quite some time when z_lib is large and n is high.
    """

    print("Fitting random projections ... ", end="")
    transformer = random_projection.SparseRandomProjection(eps=eps)
    z_lib = torch.tensor(transformer.fit_transform(z_lib))
    print("DONE")

    select_idx = 0
    last_item = z_lib[select_idx:select_idx+1]
    coreset_idx = [torch.tensor(select_idx)]
    min_distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True)

    if torch.cuda.is_available():
        last_item.to("cuda")
        z_lib.to("cuda")
        min_distances.to("cuda")

    for _ in tqdm(range(n-1)):
        distances = torch.linalg.norm(z_lib-last_item, dim=1, keepdims=True) # broadcasting step
        min_distances = torch.minimum(distances, min_distances) # iterative step
        select_idx = torch.argmax(min_distances) # selection step

        # bookkeeping
        last_item = z_lib[select_idx:select_idx+1]
        min_distances[select_idx] = 0
        coreset_idx.append(select_idx)

    return torch.stack(coreset_idx)

def write_results(results : dict, method : str):
    """Writes results to .yaml and serialized results to .txt."""
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    name = f"{method}_{timestamp}"
    with open(f"./results/{name}.yml", "w") as outfile:
        yaml.safe_dump(results, outfile, default_flow_style=False)
    with open(f"./results/{name}.txt", "w") as outfile:
        outfile.write(serialize_results(results["per_class_results"]))

def serialize_results(results : dict) -> str:
    """Serialized a results dict into something usable in markdown."""
    n_first_col = 20
    ans = []
    for k, v in results.items():
        print(v)
        s = k + " "*(n_first_col-len(k))
        s = s + f"| {v[0]*100:.1f}  | {v[1]*100:.1f}  |"
        ans.append(s)
    return "\n".join(ans)