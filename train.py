# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import os
import argparse
from tqdm import tqdm
import torch.distributed as dist
import torch.cuda.amp as amp
import pickle

from utils.data_loader import load_dataset, UnifiedCaptionDataset
from utils.tokenizer import Tokenizer
from models.caption_generator import CaptionGenerator
from models.feature_extractor import FeatureExtractor
from utils.checkpoint import save_checkpoint_async, load_checkpoint
import logging
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Image Captioning Training Script')
    parser.add_argument('--dataset', type=str, default='flickr30k', choices=['flickr8k', 'flickr30k', 'mscoco'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='data/', help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'], help='Type of RNN to use in the decoder')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size for tokenizer')
    parser.add_argument('--max_caption_length', type=int, default=20, help='Maximum caption length')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--use_cache', action='store_true', help='Use cached embeddings if available')
    args = parser.parse_args()
    return args

def setup_logging():
    logging.basicConfig(
        filename='training.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def load_cached_embeddings(dataset_name):
    cache_file = os.path.join('data', 'processed', f'{dataset_name}_cached_embeddings.pkl')
    if not os.path.exists(cache_file):
        logging.error(f"Cached embeddings file not found: {cache_file}")
        return None
    with open(cache_file, 'rb') as f:
        cached_embeddings = pickle.load(f)
    logging.info(f"Loaded cached embeddings from {cache_file}")
    return cached_embeddings

def main():
    args = parse_args()
    # Initialize distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    setup_logging()

    logging.info(f"Rank {dist.get_rank()} running on device {device}")

    # Load or build tokenizer
    tokenizer_path = os.path.join('data', 'processed', f'tokenizer_{args.dataset}.pkl')
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.load(tokenizer_path)
        logging.info(f"Tokenizer loaded from {tokenizer_path}")
    else:
        # Build tokenizer using the cleaned captions
        logging.info("Building tokenizer...")
        captions_file = os.path.join(args.data_root, 'processed', f'cleaned_captions_{args.dataset}.csv')
        df = pd.read_csv(captions_file)
        captions_list = df['caption'].tolist()
        tokenizer = Tokenizer(captions_list=captions_list, vocab_size=args.vocab_size)
        tokenizer.save(tokenizer_path)
        logging.info(f"Tokenizer saved to {tokenizer_path}")

    # Load cached embeddings if required
    cached_embeddings = None
    if args.use_cache:
        cached_embeddings = load_cached_embeddings(args.dataset)
        if cached_embeddings is None:
            logging.warning("Caching is enabled but no cached embeddings were found. Proceeding without caching.")

    # Create Dataset and DataLoader with DistributedSampler
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = load_dataset(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split='train',
        tokenizer=tokenizer,
        transform=transform,
        max_length=args.max_caption_length,
        cache_embeddings=cached_embeddings
    )
    
    sampler = DistributedSampler(dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    logging.info(f"Training dataset size: {len(train_loader.dataset)}")

    # Initialize model
    model = CaptionGenerator(
        embed_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
        vocab_size=len(tokenizer.word2idx),
        attention_dim=256,
        device=device,
        rnn_type=args.rnn_type
    ).to(device)

    # Wrap model with DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<PAD>']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Mixed Precision
    scaler = amp.GradScaler()

    # Load checkpoint if available
    checkpoint_path = os.path.join(args.checkpoint_dir, args.dataset, f'checkpoint_{args.rnn_type}_last.pth')
    if os.path.exists(checkpoint_path):
        model.module, optimizer, start_epoch = load_checkpoint(model.module, optimizer, checkpoint_path)
        logging.info(f"Resuming training from epoch {start_epoch + 1}")
    else:
        start_epoch = 0
        logging.info("Starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.num_epochs}", disable=dist.get_rank() != 0)

        for i, (features, captions, image_paths) in progress_bar:
            features = features.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            optimizer.zero_grad()

            with amp.autocast():
                outputs = model(features, captions[:, :-1], precomputed=args.use_cache)
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if dist.get_rank() == 0:
                progress_bar.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss:.4f}")
            # Save checkpoint asynchronously
            save_checkpoint_async(model.module, optimizer, epoch, avg_epoch_loss, checkpoint_dir=os.path.join(args.checkpoint_dir, args.dataset), rnn_type=args.rnn_type)

    # Clean up
    dist.destroy_process_group()
