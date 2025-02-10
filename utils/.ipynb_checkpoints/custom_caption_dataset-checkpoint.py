# utils/custom_caption_dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image

class CustomCaptionDataset(Dataset):
    def __init__(self, images_root, captions_data, transform=None):
        """
        Args:
            images_root (string): Directory with all the images.
            captions_data (list of tuples): List containing (image_id, caption) pairs.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images_root = images_root
        self.captions_data = captions_data
        self.transform = transform

        # Create a mapping from image_id to list of captions
        self.image_captions = {}
        for image_id, caption in self.captions_data:
            if image_id not in self.image_captions:
                self.image_captions[image_id] = []
            self.image_captions[image_id].append(caption)

        self.image_ids = list(self.image_captions.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_root, image_id)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        captions = self.image_captions[image_id]

        return image_id, image, captions
