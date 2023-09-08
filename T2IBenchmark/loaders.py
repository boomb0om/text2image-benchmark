import os
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from T2IBenchmark.utils import IMAGE_EXTENSIONS


class BaseImageLoader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass


class ImageDataset(BaseImageLoader, Dataset):
    def __init__(
        self,
        paths: List[str],
        preprocess_fn: Optional[Callable[[Image.Image], Any]] = None,
    ):
        self.paths = paths
        self.preprocess_fn = preprocess_fn if preprocess_fn else lambda x: x

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any:
        image = Image.open(self.paths[idx])
        preproc = self.preprocess_fn(image)
        return preproc
    
    def __str__(self) -> str:
        return f"ImageDataset({self.__len__()} items)"


class CaptionImageDataset(Dataset):
    def __init__(
        self,
        images_paths: List[str],
        captions: List[str],
        preprocess_fn: Optional[Callable[[Image.Image], Any]] = None,
    ):
        assert len(images_paths) == len(captions)
        self.images_paths = images_paths
        self.captions = captions
        self.preprocess_fn = preprocess_fn if preprocess_fn else lambda x: x

    def __len__(self) -> int:
        return len(self.images_paths)
    
    def __getitem__(self, idx: int) -> tuple:
        image = Image.open(self.images_paths[idx])
        return self.preprocess_fn(image), self.captions[idx]
    
    def __str__(self) -> str:
        return f"CaptionImageDataset({self.__len__()} items)"


def get_images_from_folder(folder_path: str) -> ImageDataset:
    filepaths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1][1:]
            if ext in IMAGE_EXTENSIONS:
                filepath = os.path.join(root, file)
                filepaths.append(filepath)

    return filepaths


def validate_image_paths(paths: List[str]) -> bool:
    for path in paths:
        file = os.path.basename(path)
        ext = os.path.splitext(file)[1][1:]
        assert os.path.exists(path), f"File {path} is not exists"
        assert ext in IMAGE_EXTENSIONS, f"File {path} is not an Image"
    return True
