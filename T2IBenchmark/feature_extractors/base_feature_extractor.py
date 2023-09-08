from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import torch


class BaseFeatureExtractor(ABC):
    """
    A base class for feature extraction methods.

    This class serves as an interface for feature extraction techniques
    and should be subclassed for specific implementations, such as InceptionV3FE.
    """

    @abstractmethod
    def get_preprocess_fn(self) -> Callable[[Image.Image], np.ndarray]:
        """
        Get the preprocessing function for the input images.

        This function should be implemented by the subclass and should
        define the specific preprocessing steps needed for the feature
        extractor.

        Returns
        -------
        Callable[[Image.Image], np.ndarray]
            The preprocessing function that takes an input PIL.Image.Image and
            returns a preprocessed numpy array.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass for the feature extractor.

        This function should be implemented by the subclass and should
        define the forward pass logic for the feature extractor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to process.

        Returns
        -------
        torch.Tensor
            The output tensor with the extracted features.
        """
        pass
