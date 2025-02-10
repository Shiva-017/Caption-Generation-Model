# models/llm_caption_generator.py

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, UnidentifiedImageError

class LLMCaptionGenerator:
    def __init__(self, device='cuda'):
        """
        LLM-based Caption Generator using Hugging Face's BLIP model.

        Args:
            device (str): Device to run the model on.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        self.model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(self.device)
        self.model.eval()

    def generate_caption(self, image_path):
        """
        Generate a caption for the given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Generated caption or "N/A" if image cannot be processed.
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, UnidentifiedImageError):
            print(f"Cannot process image: {image_path}")
            return "N/A"

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
