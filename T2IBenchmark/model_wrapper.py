from typing import List, Dict, Union, Optional, Callable, Iterable, Any
from abc import ABC, abstractmethod
import os
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset
try:
    from torch.utils.data import default_collate
except ImportError:
    from torch.utils.data._utils.collate import default_collate

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
        device: torch.device,
        save_dir: Optional[str] = None,
        use_saved_images: bool = False,
        seed: Optional[int] = 42,
        output_format: str = 'JPEG',
        jpeg_quality: int = 90
    ):
        assert output_format in OUTPUT_FORMATS
        
        self.captions = []
        self.file_ids = None
        
        self.device = device
        self.seed = seed
        self.use_saved_images = use_saved_images
        self.save_dir = save_dir
        self.output_format = output_format
        self.jpeg_quality = jpeg_quality
        
        self.load_model(self.device)
        
    def set_captions(self, captions: List[str], file_ids: Optional[List[int]] = None):
        self.captions = captions
        self.file_ids = file_ids
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            if self.file_ids is None:
                self.file_ids = [i for i in range(len(captions))]
            assert len(self.file_ids) == len(self.captions)
        
    @abstractmethod
    def load_model(self, device: torch.device):
        """Initialize model here"""
        
    @abstractmethod
    def generate(self, caption: str) -> Image.Image:
        """Generate PIL image for provided caption"""
        
    def __len__(self) -> int:
        return len(self.captions)
    
    def _get_filepath(self, idx: int) -> Union[None, str]:
        filepath = None
        if self.save_dir:
            filename = str(self.file_ids[idx]) + '.' + self.output_format.lower()
            filepath = os.path.join(self.save_dir, filename)
        return filepath
    
    def _get_saved(self, idx: int) -> Union[None, Image.Image]:
        filepath = self._get_filepath(idx)
        if os.path.exists(filepath):
            return Image.open(filepath)

    def __getitem__(self, idx: int) -> Image.Image:
        if self.seed:
            set_all_seeds(self.seed)
            
        caption = self.captions[idx]
        
        if self.save_dir and self.use_saved_images:
            cached_img = self._get_saved(idx)
            if cached_img:
                return cached_img
        image = self.generate(caption)
        
        if self.save_dir or self.output_format == 'JPEG':
            buf = save_img_to_buffer(image, self.output_format, self.jpeg_quality)
        if self.save_dir:
            filepath = self._get_filepath(idx)
            with open(filepath, 'wb') as f:
                f.write(buf.getbuffer())
            buf.seek(0)
            
        if self.output_format == 'JPEG':
            image = Image.open(buf)
        
        return image
    
    
class ModelWrapperDataloader:
    
    def __init__(
        self, 
        model_wrapper: T2IModelWrapper,
        batch_size: int = 1,
        preprocess_fn: Optional[Callable[[Image.Image], Any]] = None,
        collate_fn: Callable[[list], torch.Tensor] = default_collate,
    ):
        self.model_wrapper = model_wrapper
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn if preprocess_fn else lambda x: x
        self.collate_fn = collate_fn
    
    def __len__(self) -> int:
        main_batches = len(self.model_wrapper)//self.batch_size
        if len(self.model_wrapper)%self.batch_size > 0:
            main_batches += 1
        return main_batches
    
    def __iter__(self) -> Iterable[torch.Tensor]:
        total_items = len(self.model_wrapper)
        
        for c in range(0, total_items, self.batch_size):
            if total_items-c >= self.batch_size:
                bs = self.batch_size
            elif total_items-c > 0:
                bs = total_items-c
            else:
                break

            samples = []
            for i in range(bs):
                sample_num = c+i
                generated_image = self.model_wrapper[sample_num]
                generated_image = generated_image.convert('RGB')
                preprocessed_image = self.preprocess_fn(generated_image)
                samples.append(preprocessed_image)
            yield self.collate_fn(samples)
