from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import torch


class BaseFeatureExtractor(ABC):

    @abstractmethod
    def get_preprocess_fn(self) -> Callable[[Image.Image], np.ndarray]:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
