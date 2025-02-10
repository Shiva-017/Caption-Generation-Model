# utils/visualization.py

import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np

def visualize_attention(image, caption, attention_weights, tokenizer, save_path=None):
    """
    Visualize attention weights on the image for the given caption.

    Args:
        image (PIL.Image): Image to visualize.
        caption (str): Generated caption.
        attention_weights (Tensor): Attention weights (num_words, height, width).
        tokenizer (Tokenizer): Tokenizer instance.
        save_path (str, optional): Path to save the visualization.
    """
    num_words = attention_weights.size(0)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis('off')

    for t in range(num_words):
        if t >= len(caption.split()):
            break
        word = caption.split()[t]
        attn = attention_weights[t].cpu().numpy()

        # Verify attention weights shape
        if attn.ndim != 2:
            print(f"Attention weights for word '{word}' have incorrect shape: {attn.shape}. Expected 2D.")
            continue

        attn = attn / attn.max()
        attn_img = Image.fromarray((attn * 255).astype('uint8')).resize(image.size, Image.BILINEAR)
        attn_img = attn_img.convert("RGBA")
        attn_overlay = Image.new("RGBA", image.size)
        attn_overlay = Image.alpha_composite(attn_overlay, attn_img)
        ax.imshow(attn_overlay, alpha=0.4)

    plt.title(f"Caption: {caption}", fontsize=12)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def show_caption_comparison(image, reference, hypothesis, llm_caption, save_path=None):
    """
    Display side-by-side comparison of captions.

    Args:
        image (PIL.Image): Image being captioned.
        reference (str): Ground truth caption.
        hypothesis (str): Model-generated caption.
        llm_caption (str): LLM-generated caption.
        save_path (str, optional): Path to save the visualization.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    plt.axis('off')

    # Add text boxes below the image
    plt.figtext(0.1, 0.05, f"Reference: {reference}", wrap=True, horizontalalignment='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.figtext(0.1, 0.02, f"Model: {hypothesis}", wrap=True, horizontalalignment='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.figtext(0.1, 0.00, f"LLM: {llm_caption}", wrap=True, horizontalalignment='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def visualize_embedding_space(embeddings, labels, save_path=None):
    """
    Visualize embedding spaces using PCA or t-SNE.

    Args:
        embeddings (Tensor): Embedding vectors (num_samples, dim).
        labels (list): Labels or categories for each sample.
        save_path (str, optional): Path to save the visualization.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings.cpu().numpy())

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=labels, palette='viridis', alpha=0.7)
    plt.title('Embedding Space Visualization (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
