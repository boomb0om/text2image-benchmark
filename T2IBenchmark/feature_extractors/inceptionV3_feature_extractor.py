from typing import Callable
import numpy as np
import torch
import torchvision
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

    def get_preprocess_fn(self) -> Callable[[Image.Image], torch.Tensor]:
        resizer = self.resizer
        transforms = torchvision.transforms.ToTensor()

        def preprocess(image: Image.Image) -> np.ndarray:
            image = image.convert('RGB')
            image_resized = resizer(np.array(image))
            x = transforms(image_resized)
            return 2 * x - 1

        return preprocess

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device)
            feat = self.inception(x)[0].squeeze(-1).squeeze(-1)
        return feat.detach().cpu()
