import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from PIL import Image
from torch import tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATASETS_PATH = Path("./datasets")
IMAGENET_MEAN = tensor([0.485, 0.456, 0.406])
IMAGENET_STD = tensor([0.229, 0.224, 0.225])

MVTEC_CLASSES = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz",
}


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract regular files and directories without allowing path traversal."""
    destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if not member_path.is_relative_to(destination):
            raise ValueError(f"Unsafe archive member: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsupported archive member: {member.name}")
    archive.extractall(destination)


class MVTecDataset:
    def __init__(
        self, class_name: str, size: int = 224, root: str | Path = DATASETS_PATH
    ):
        self.class_name = class_name
        self.size = size
        self.root = Path(root)
        if class_name in MVTEC_CLASSES:
            self._download(MVTEC_CLASSES[class_name])
        elif not (self.root / class_name).is_dir():
            raise FileNotFoundError(
                f"Dataset '{class_name}' was not found under '{self.root}'"
            )
        self.train_ds = MVTecTrainDataset(class_name, size, self.root)
        self.test_ds = MVTecTestDataset(class_name, size, self.root)

    def _download(self, url: str):
        dataset_path = self.root / self.class_name
        if dataset_path.is_dir():
            print(f"   Found '{self.class_name}' in '{self.root}/'\n")
            return

        self.root.mkdir(parents=True, exist_ok=True)
        print(
            f"   Could not find '{self.class_name}' in '{self.root}/'. Downloading ..."
        )
        with TemporaryDirectory(prefix=".mvtec-download-", dir=self.root) as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / f"{self.class_name}.tar.xz"
            extraction_path = temp_path / "extracted"
            extraction_path.mkdir()
            urlretrieve(url, archive_path)
            with tarfile.open(archive_path, mode="r:xz") as archive:
                _safe_extract(archive, extraction_path)

            extracted_dataset = extraction_path / self.class_name
            if not extracted_dataset.is_dir():
                raise ValueError(
                    f"Downloaded archive does not contain '{self.class_name}'"
                )
            if not dataset_path.exists():
                extracted_dataset.rename(dataset_path)
        print("")

    def get_datasets(self):
        return self.train_ds, self.test_ds

    def get_dataloaders(self):
        return DataLoader(self.train_ds), DataLoader(self.test_ds)


class MVTecTrainDataset(ImageFolder):
    def __init__(self, class_name: str, size: int, root: str | Path = DATASETS_PATH):
        self.dataset_root = Path(root) / class_name
        super().__init__(
            root=self.dataset_root / "train",
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            ),
        )
        self.class_name = class_name
        self.size = size


class MVTecTestDataset(ImageFolder):
    def __init__(self, class_name: str, size: int, root: str | Path = DATASETS_PATH):
        self.dataset_root = Path(root) / class_name
        self.test_root = self.dataset_root / "test"
        super().__init__(
            root=self.test_root,
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            ),
            target_transform=transforms.Compose(
                [
                    transforms.Resize(
                        256, interpolation=transforms.InterpolationMode.NEAREST
                    ),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                ]
            ),
        )
        self.class_name = class_name
        self.size = size

    def __getitem__(self, index):
        path = Path(self.samples[index][0])
        sample = self.loader(str(path))

        if path.parent.name == "good":
            target = Image.new("L", (self.size, self.size), color=0)
            sample_class = 0
        else:
            relative_path = path.relative_to(self.test_root)
            target_path = (
                self.dataset_root
                / "ground_truth"
                / relative_path.parent
                / f"{relative_path.stem}_mask.png"
            )
            if not target_path.is_file():
                raise FileNotFoundError(f"Missing anomaly mask: {target_path}")
            with Image.open(target_path) as target_image:
                target = target_image.convert("L")
            sample_class = 1

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target[:1], sample_class


class StreamingDataset:
    """In-memory image adapter for interactive inspection workflows."""

    def __init__(self, size: int = 224):
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.samples = []

    def add_pil_image(self, image: Image.Image):
        image = image.convert("RGB")
        self.samples.append(image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (self.transform(sample), tensor(0.0))
