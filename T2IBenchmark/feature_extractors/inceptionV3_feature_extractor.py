from typing import Callable
import numpy as np
import torch
from PIL import Image

from T2IBenchmark.feature_extractors.base_feature_extractor import BaseFeatureExtractor
from T2IBenchmark.feature_extractors.inceptionV3 import InceptionV3
from T2IBenchmark.utils import build_resizer


class InceptionV3FE(BaseFeatureExtractor):
    """Pretrained InceptionV3 feature extractor"""

    def __init__(self, device: torch.device):
        self.device = device

        self.inception = InceptionV3()
        self.inception.to(self.device)
        self.inception.eval()

        self.resizer = build_resizer("clean")

    def get_preprocess_fn(self) -> Callable[[Image.Image], np.ndarray]:
        resizer = self.resizer

        def preprocess(image: Image.Image) -> np.ndarray:
            image_resized = resizer(np.array(image))
            return image_resized

        return preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.inception(x.to(self.device))
        return feat.detach().cpu()
