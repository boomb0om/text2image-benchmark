from typing import List, Optional, Callable, Any
from abc import abstractmethod
from PIL import Image
from torch.utils.data import Dataset


class BaseImageLoader:

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
        preprocess_fn: Optional[Callable[[Image.Image], Any]] = None
    ):
        self.paths = paths
        self.preprocess_fn = preprocess_fn if preprocess_fn else lambda x: x

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any:
        image = Image.open(self.paths[idx])
        preproc = self.preprocess_fn(image)
        return preproc
