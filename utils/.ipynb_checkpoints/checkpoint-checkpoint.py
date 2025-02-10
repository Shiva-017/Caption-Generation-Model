# utils/checkpoint.py

import torch
import os
import logging
import hashlib

def setup_logger(log_file='checkpoint.log'):
    """
    Set up logging configuration.

    Args:
        log_file (str): File to save logs.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, rnn_type='LSTM'):
    """
    Save model and optimizer states.

    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer to save.
        epoch (int): Current epoch.
        loss (float): Current loss.
        checkpoint_dir (str): Directory to save the checkpoint.
        rnn_type (str): Type of RNN used ('LSTM' or 'GRU').
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{rnn_type}_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")

def save_checkpoint_async(model, optimizer, epoch, loss, checkpoint_dir, rnn_type='LSTM'):
    """
    Asynchronously save checkpoint to avoid blocking training.

    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer to save.
        epoch (int): Current epoch.
        loss (float): Current loss.
        checkpoint_dir (str): Directory to save the checkpoint.
        rnn_type (str): Type of RNN used ('LSTM' or 'GRU').
    """
    import threading
    thread = threading.Thread(target=save_checkpoint, args=(model, optimizer, epoch, loss, checkpoint_dir, rnn_type))
    thread.start()

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model and optimizer states from checkpoint.

    Args:
        model (nn.Module): Model to load state into.
        optimizer (Optimizer): Optimizer to load state into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        tuple: Loaded model, optimizer, epoch
    """
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Verify integrity using checksum
    checksum = calculate_checksum(checkpoint_path)
    logging.info(f"Loading checkpoint from {checkpoint_path} with checksum {checksum}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    logging.info(f"Checkpoint loaded: Epoch {epoch}, Loss {loss}")
    return model, optimizer, epoch

def calculate_checksum(file_path):
    """
    Calculate MD5 checksum of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: MD5 checksum.
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()
