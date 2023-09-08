from typing import List, Dict, Optional, Callable, Any
from abc import ABC, abstractmethod
import os
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset

from T2IBenchmark.loaders import BaseImageLoader
from T2IBenchmark.utils import IMAGE_EXTENSIONS, set_all_seeds


OUTPUT_FORMATS = {'PNG', 'JPEG'}


def save_img_to_buffer(img: Image.Image, output_format: str, compression: int) -> BytesIO:
    buf = BytesIO()
    if output_format == 'PNG':
        img.save(buf, format=output_format)
    else:
        img.save(buf, format=output_format, quality=compression)
    return buf


class T2IModelWrapper(BaseImageLoader):
    
    def __init__(
        self, 
        captions: List[str],
        device: torch.device,
        save_dir: Optional[str] = None,
        file_ids: Optional[List[int]] = None, 
        seed: Optional[int] = 42,
        output_format: str = 'JPEG',
        jpeg_quality: int = 90
    ):
        assert output_format in OUTPUT_FORMATS
        
        self.captions = captions
        self.device = device
        self.seed = seed
        
        self.save_dir = save_dir
        self.file_ids = file_ids
        if self.save_dir:
            if self.file_ids is None:
                self.file_ids = [i for i in range(len(captions))]
            assert len(self.file_ids) == len(self.captions)

        self.output_format = output_format
        self.jpeg_quality = jpeg_quality
        
        self.load_model(self.device)
       
    @abstractmethod
    def load_model(self, device: torch.device):
        """Initialize model here"""
        
    @abstractmethod
    def generate(self, caption: str) -> Image.Image:
        """Generate PIL image for provided caption"""
        
    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Image.Image:
        if self.seed:
            set_all_seeds(self.seed)
            
        caption = self.captions[idx]
        image = self.generate(caption)
        
        if self.save_dir or self.output_format == 'JPEG':
            buf = save_img_to_buffer(image, self.output_format, self.jpeg_quality)
        if self.save_dir:
            filename = str(self.file_ids[idx]) + '.' + self.output_format.lower()
            filepath = os.path.join(self.save_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(buf.getbuffer())
            buf.seek(0)
            
        if self.output_format == 'JPEG':
            image = Image.open(buf)
        
        return image