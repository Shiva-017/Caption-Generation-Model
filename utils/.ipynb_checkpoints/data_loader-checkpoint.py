# utils/data_loader.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import pandas as pd
import logging

from utils.tokenizer import Tokenizer

class UnifiedCaptionDataset(Dataset):
    def __init__(self, dataset_name, root_dir, split, tokenizer, transform=None, max_length=20, cache_embeddings=None):
        """
        Unified Dataset for Flickr8k, Flickr30k, and MSCOCO.

        Args:
            dataset_name (str): Name of the dataset ('flickr8k', 'flickr30k', 'mscoco').
            root_dir (str): Root directory where the dataset is stored.
            split (str): Split of the dataset ('train', 'val', 'test').
            tokenizer (Tokenizer): Tokenizer instance.
            transform (callable, optional): Optional transform to be applied on an image.
            max_length (int): Maximum caption length.
            cache_embeddings (dict, optional): Precomputed image embeddings.
        """
        self.dataset_name = dataset_name.lower()
        self.root_dir = root_dir
        self.split = split.lower()
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.cache_embeddings = cache_embeddings

        if self.dataset_name == 'flickr30k':
            captions_file = os.path.join(root_dir, 'processed', f'cleaned_captions_{dataset_name}.csv')
            images_dir = os.path.join(root_dir, dataset_name, 'flickr30k_images')
            self.dataset = self._load_flickr30k(captions_file, images_dir)
        elif self.dataset_name == 'flickr8k':
            # Existing code for Flickr8k
            pass
        elif self.dataset_name == 'mscoco':
            # Existing code for MSCOCO
            pass
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _load_flickr30k(self, captions_file, images_dir):
        df = pd.read_csv(captions_file)
        # Filter by split
        df = df[df['split'] == self.split]
        df['image_path'] = df['image'].apply(lambda x: os.path.join(images_dir, x))
        df = df[df['image_path'].apply(os.path.exists)]
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        record = self.dataset.iloc[idx]
        image_path = record['image_path']
        caption = record['caption']

        if self.cache_embeddings and image_path in self.cache_embeddings:
            features = torch.tensor(self.cache_embeddings[image_path], dtype=torch.float)
        else:
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            except (FileNotFoundError, UnidentifiedImageError):
                # Return a tensor of zeros and an empty caption if image is corrupted
                image = Image.new('RGB', (224, 224), (0, 0, 0))
                if self.transform:
                    image = self.transform(image)
            # Feature extraction is handled outside if caching is not used
            features = image  # Assuming transform includes tensor conversion

        # Encode caption
        caption_indices = self.tokenizer.encode(caption, self.max_length)
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)

        return features, caption_tensor, image_path

def load_dataset(dataset_name, root_dir, split, tokenizer, transform=None, max_length=20, cache_embeddings=None):
    """
    Initialize and return a UnifiedCaptionDataset.

    Args:
        dataset_name (str): Name of the dataset ('flickr8k', 'flickr30k', 'mscoco').
        root_dir (str): Root directory of the dataset.
        split (str): Dataset split ('train', 'val', 'test').
        tokenizer (Tokenizer): Tokenizer instance.
        transform (callable, optional): Transformations to apply to images.
        max_length (int): Maximum caption length.
        cache_embeddings (dict, optional): Precomputed image embeddings.

    Returns:
        UnifiedCaptionDataset: Initialized dataset.
    """
    return UnifiedCaptionDataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        split=split,
        tokenizer=tokenizer,
        transform=transform,
        max_length=max_length,
        cache_embeddings=cache_embeddings
    )
