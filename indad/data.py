import tarfile
import wget
from pathlib import Path
import numpy as np
from PIL import Image

from torchvision.datasets import ImageFolder
from torchvision import transforms

import os
from os.path import isdir

DATASETS_PATH = Path("./datasets")

def mvtec_classes():
    return [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

class MVTecDataset:
    def __init__(self, cls : str, size : int = 224):
        self.cls = cls
        self.size = size
        if cls in mvtec_classes():
            self._download()
        self.train_ds = MVTecTrainDataset(cls, size)
        self.val_ds = MVTecTestDataset(cls, size)

    def _download(self):
        if not isdir(DATASETS_PATH / self.cls):
            print(f"Could not find '{self.cls}' in '{DATASETS_PATH}/'. Downloading ... ")
            url = f"ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/{self.cls}.tar.xz"
            wget.download(url)
            with tarfile.open(f"{self.cls}.tar.xz") as tar:
                tar.extractall(DATASETS_PATH)
            os.remove(f"{self.cls}.tar.xz")
            print("") # force newline

    def load(self):
        return self.train_ds, self.val_ds

class MVTecTrainDataset(ImageFolder):
    def __init__(self, cls : str, size : int):
        super().__init__(
            root=DATASETS_PATH / cls / "train",
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ])
        )
        self.cls = cls
        self.size = size

class MVTecTestDataset(ImageFolder):
    def __init__(self, cls : str, size : int):
        super().__init__(
            root=DATASETS_PATH / cls / "test",
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]),
            target_transform=transforms.Compose([
                transforms.Resize(
                    256,
                    interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]),
        )
        self.cls = cls
        self.size = size
            
    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = self.loader(path)
        
        if "good" in path:
            target = Image.new('L', (self.size, self.size))
            sample_class = 0
        else:
            target_path = path.replace("test", "ground_truth")
            target_path = target_path.replace(".png", "_mask.png")
            target = self.loader(target_path)
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target[:1], sample_class
