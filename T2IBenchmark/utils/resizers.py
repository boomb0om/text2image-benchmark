from typing import Tuple
import numpy as np
import cv2
from PIL import Image


NAME2PIL_FILTER = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
    "lanczos": Image.LANCZOS,
    "box": Image.BOX
}

NAME2CV2_FILTER = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA
}


def resize_single_channel(x_np: np.ndarray, output_size: Tuple[int, int], filter) -> np.ndarray:
    s1, s2 = output_size
    img = Image.fromarray(x_np.astype(np.float32), mode='F')
    img = img.resize(output_size, resample=filter)
    return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)


class Resizer:

    def __init__(self, lib: str, filter_name: str, quantize_after: bool, output_size: Tuple[int, int]):
        assert lib in ['PIL', 'OpenCV']

        self.lib = lib
        self.filter_name = filter_name
        self.quantize_after = quantize_after
        self.output_size = output_size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.lib == "PIL" and self.quantize_after:
            filter_ = NAME2PIL_FILTER[self.filter_name]
            x = Image.fromarray(img)
            x = x.resize(self.output_size, resample=filter_)
            x = np.asarray(x).clip(0, 255).astype(np.uint8)
        elif self.lib == "PIL" and not self.quantize_after:
            filter_ = NAME2PIL_FILTER[self.filter_name]
            x = [resize_single_channel(img[:, :, idx], self.output_size, filter_) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
        elif self.lib == "OpenCV":
            filter_ = NAME2CV2_FILTER[self.filter_name]
            x = cv2.resize(img, self.output_size, interpolation=filter_)
            x = x.clip(0, 255)
            if self.quantize_after:
                x = x.astype(np.uint8)
        return x


def build_resizer(mode: str) -> Resizer:
    if mode == "clean":
        return Resizer("PIL", "bicubic", False, (299, 299))
    else:
        raise ValueError(f"Invalid resize mode: {mode}")
