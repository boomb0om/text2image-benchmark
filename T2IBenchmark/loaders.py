from typing import List, Optional, Callable, Any
from abc import ABC, abstractmethod
import os
from PIL import Image
from torch.utils.data import Dataset

from T2IBenchmark.utils import IMAGE_EXTENSIONS


class BaseImageLoader(ABC):
    """
    A base class for custom image loader implementations.

    This class serves as an interface for various image loading techniques
    and should be subclassed for specific custom implementations.
    """
    @abstractmethod
    def __len__(self) -> int:
        """
       This method should be implemented by the subclass and should return
       the total number of images in the loader.

       Returns
       -------
       int
           The total number of images or samples in the loader.
       """
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        """
        Retrieve the image at the specified index.

        This method should be implemented by the subclass and should return
        the image at a given index with any necessary processing
        (e.g., resizing, normalization) already applied.

        Parameters
        ----------
        idx : int
            The index of the image or sample to retrieve.

        Returns
        -------
        Any
            The image at the specified index, processing as needed by
            the subclass implementation.
        """
        pass


class ImageDataset(BaseImageLoader, Dataset):
    """
    An image dataset loader for managing and preprocessing image data.

    This class inherits from BaseImageLoader and Dataset to provide an
    interface for loading and manipulating images, including optional
    preprocessing steps.

    Attributes
    ----------
    paths : List[str]
       A list of file paths to the images.
    preprocess_fn : Optional[Callable[[Image.Image], Any]], optional
       An optional preprocessing function that takes a PIL.Image.Image
       and applies preprocessing steps before returning the processed image.
    """
    def __init__(
        self,
        paths: List[str],
        preprocess_fn: Optional[Callable[[Image.Image], Any]] = None
    ):
        """
        Initialize the ImageDataset with the list of image paths and an optional preprocessing function.

        Parameters
        ----------
        paths : List[str]
            A list of file paths to the images.
        preprocess_fn : Optional[Callable[[Image.Image], Any]], optional
            An optional preprocessing function that takes a PIL.Image.Image
            and applies preprocessing steps before returning the processed image, by default None.
        """
        self.paths = paths
        self.preprocess_fn = preprocess_fn if preprocess_fn else lambda x: x

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns
        -------
        int
            The total number of images in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, idx: int) -> Any:
        """
        Retrieve and preprocess the image at the specified index.

        Parameters
        ----------
        idx : int
            The index of the image to retrieve.

        Returns
        -------
        Any
            The preprocessed image at the specified index.
        """
        image = Image.open(self.paths[idx])
        preproc = self.preprocess_fn(image)
        return preproc
    
    def __str__(self) -> str:
        """
        Returns a string representation of the ImageDataset, showing the total number of items.

        Returns
        -------
        str
            The string representation of the ImageDataset.
        """
        return f"ImageDataset({self.__len__()} items)"
    
    
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