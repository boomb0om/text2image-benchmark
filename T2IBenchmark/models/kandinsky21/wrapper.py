import torch
from PIL import Image
from T2IBenchmark import T2IModelWrapper
from kandinsky2 import get_kandinsky2


class Kandinsky21Wrapper(T2IModelWrapper):
    
    def load_model(self, device: torch.device):
        """Initialize model here"""
        self.model = get_kandinsky2(device, task_type='text2img', model_version='2.1', use_flash_attention=False)
        
    def generate(self, caption: str) -> Image.Image:
        """Generate PIL image for provided caption"""
        images = self.model.generate_text2img(
            caption, 
            num_steps=75,
            batch_size=1, 
            guidance_scale=4,
            h=768, w=768,
            sampler='p_sampler', 
            prior_cf_scale=4,
            prior_steps="5"
        )
        return images[0]
