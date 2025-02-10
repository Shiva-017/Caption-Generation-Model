# cache_embeddings.py

import torch
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import pickle

from utils.data_loader import load_dataset
from utils.tokenizer import Tokenizer
from models.feature_extractor import FeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description='Cache Image Embeddings')
    parser.add_argument('--dataset', type=str, required=True, choices=['flickr8k', 'flickr30k', 'mscoco'], help='Dataset to cache')
    parser.add_argument('--data_root', type=str, default='data/', help='Root directory of the dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size')
    parser.add_argument('--max_length', type=int, default=20, help='Maximum caption length')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='Directory to save cached embeddings')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer_path = os.path.join(args.data_root, 'processed', f'tokenizer_{args.dataset}.pkl')
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer file not found at {tokenizer_path}. Please ensure it exists.")
        return
    tokenizer = Tokenizer.load(tokenizer_path)

    # Initialize Feature Extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor = FeatureExtractor(model_name='resnet50', fine_tune=False, device=device)
    feature_extractor.eval()

    # Create Dataset and DataLoader
    dataset = load_dataset(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split=args.split,
        tokenizer=tokenizer,
        transform=None,
        max_length=args.max_length,
        cache_embeddings=None  # No caching during initial feature extraction
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Cache embeddings
    cached_embeddings = {}
    with torch.no_grad():
        for images, captions, image_paths in tqdm(loader, desc="Caching Embeddings"):
            images = images.to(device)
            features = feature_extractor(images)
            features = features.cpu()
            for feature, image_path in zip(features, image_paths):
                cached_embeddings[image_path] = feature.numpy()

    # Save cached embeddings
    cache_file = os.path.join(args.output_dir, f'{args.dataset}_cached_embeddings.pkl')
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_embeddings, f)
    print(f"Cached embeddings saved to {cache_file}")

if __name__ == '__main__':
    main()
