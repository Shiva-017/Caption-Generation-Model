# evaluate.py

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
from PIL import Image

from utils.data_loader import load_dataset, UnifiedCaptionDataset
from utils.tokenizer import Tokenizer
from models.caption_generator import CaptionGenerator
from models.llm_caption_generator import LLMCaptionGenerator
from utils.evaluation import compute_metrics
from utils.visualization import visualize_attention, show_caption_comparison
from utils.checkpoint import load_checkpoint
import logging
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Image Captioning Evaluation Script')
    parser.add_argument('--dataset', type=str, default='flickr30k', choices=['flickr8k', 'flickr30k', 'mscoco'], help='Dataset to evaluate')
    parser.add_argument('--data_root', type=str, default='data/', help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_LSTM_last.pth', help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples for visualization')
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'], help='Type of RNN used in the model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation')
    parser.add_argument('--use_cache', action='store_true', help='Use cached embeddings if available')
    args = parser.parse_args()
    return args

def setup_logging():
    logging.basicConfig(
        filename='evaluation.log',
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
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    setup_logging()
    logging.info(f"Evaluating on device: {device}")

    # Load tokenizer
    tokenizer_path = os.path.join('data', 'processed', f'tokenizer_{args.dataset}.pkl')
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer file not found at {tokenizer_path}. Please ensure it exists.")
        return
    tokenizer = Tokenizer.load(tokenizer_path)
    logging.info(f"Tokenizer loaded from {tokenizer_path}")


    # Load cached embeddings if required
    cached_embeddings = None
    if args.use_cache:
        cached_embeddings = load_cached_embeddings(args.dataset)
        if cached_embeddings is None:
            logging.warning("Caching is enabled but no cached embeddings were found. Proceeding without caching.")

    # Create DataLoader for test split
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = load_dataset(
        dataset_name=args.dataset,
        root_dir=args.data_root,
        split='test',
        tokenizer=tokenizer,
        transform=transform,
        max_length=20,
        cache_embeddings=cached_embeddings
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    logging.info(f"Total test samples: {len(test_loader.dataset)}")

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

    # Load checkpoint
    checkpoint_path = os.path.join('checkpoints', args.dataset, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint file not found at {checkpoint_path}. Please ensure it exists.")
        return
    model, _, _ = load_checkpoint(model, None, checkpoint_path)
    model.eval()
    logging.info(f"Loaded checkpoint from {checkpoint_path}")

    # Initialize LLM Caption Generator
    llm_caption_generator = LLMCaptionGenerator(device=device)

    # Evaluation metrics containers
    references = []
    hypotheses = []
    llm_hypotheses = []

    with torch.no_grad():
        for batch_idx, (features, captions, image_paths) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if features is None or captions is None:
                # Skip this batch
                logging.warning(f"Skipped batch {batch_idx} due to missing features or captions.")
                continue

            # Generate captions using your model
            outputs, alphas = model.generate(features, tokenizer, max_length=20, precomputed=args.use_cache)

            # Decode captions
            for i in range(features.size(0)):
                reference = tokenizer.decode(captions[i].cpu().numpy())
                hypothesis = tokenizer.decode(outputs[i].cpu().numpy())

                # Get image path
                image_path = image_paths[i]
                if os.path.exists(image_path):
                    llm_caption = llm_caption_generator.generate_caption(image_path)
                else:
                    llm_caption = "N/A"

                references.append(reference)
                hypotheses.append(hypothesis)
                llm_hypotheses.append(llm_caption)

                # Visualization
                if len(references) <= args.num_samples and llm_caption != "N/A":
                    visualize_sample(image_path, reference, hypothesis, llm_caption, alphas[i], tokenizer)

    # Compute Metrics
    bleu_scores, cider_score, meteor_score, rouge_score, semantic_similarity = compute_metrics(references, hypotheses)
    bleu_scores_llm, cider_score_llm, meteor_score_llm, rouge_score_llm, semantic_similarity_llm = compute_metrics(references, llm_hypotheses)

    # Display Metrics
    logging.info("\nYour Model Evaluation Metrics:")
    logging.info(f"BLEU Scores: {bleu_scores}")
    logging.info(f"CIDEr Score: {cider_score}")
    logging.info(f"METEOR Score: {meteor_score}")
    logging.info(f"ROUGE Score: {rouge_score}")
    logging.info(f"Semantic Similarity: {semantic_similarity}")

    logging.info("\nLLM Model Evaluation Metrics:")
    logging.info(f"BLEU Scores: {bleu_scores_llm}")
    logging.info(f"CIDEr Score: {cider_score_llm}")
    logging.info(f"METEOR Score: {meteor_score_llm}")
    logging.info(f"ROUGE Score: {rouge_score_llm}")
    logging.info(f"Semantic Similarity: {semantic_similarity_llm}")

    # Optionally, print metrics to console
    print("\nYour Model Evaluation Metrics:")
    print(f"BLEU Scores: {bleu_scores}")
    print(f"CIDEr Score: {cider_score}")
    print(f"METEOR Score: {meteor_score}")
    print(f"ROUGE Score: {rouge_score}")
    print(f"Semantic Similarity: {semantic_similarity}")

    print("\nLLM Model Evaluation Metrics:")
    print(f"BLEU Scores: {bleu_scores_llm}")
    print(f"CIDEr Score: {cider_score_llm}")
    print(f"METEOR Score: {meteor_score_llm}")
    print(f"ROUGE Score: {rouge_score_llm}")
    print(f"Semantic Similarity: {semantic_similarity_llm}")

def visualize_sample(image_path, reference, hypothesis, llm_caption, alphas, tokenizer):
    from PIL import Image

    image = Image.open(image_path).convert('RGB')

    # Visualize attention for your model
    visualize_attention(image, hypothesis, alphas, tokenizer)

    # Show caption comparison between reference, your model, and LLM
    show_caption_comparison(image, reference, hypothesis, llm_caption)

if __name__ == '__main__':
    main()
