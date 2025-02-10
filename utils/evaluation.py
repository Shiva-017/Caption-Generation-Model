# utils/evaluation.py

import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
from collections import defaultdict

# Handle pycocoevalcap imports gracefully
try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
except ImportError:
    raise ImportError("Please install pycocoevalcap from https://github.com/tylin/coco-caption.git")

# Initialize global variable for word vectors
word_vectors = None

def load_word_vectors():
    """
    Load pre-trained word embeddings (e.g., GloVe).

    Returns:
        gensim.models.KeyedVectors: Loaded word vectors.
    """
    global word_vectors
    if word_vectors is None:
        print("Loading word embeddings...")
        word_vectors = api.load("glove-wiki-gigaword-100")  # 100-dimensional embeddings
    return word_vectors

def get_sentence_embedding(sentence):
    """
    Compute the average word embedding for a sentence.

    Args:
        sentence (str): The sentence to embed.

    Returns:
        np.array: Average embedding vector.
    """
    word_vectors = load_word_vectors()
    words = sentence.lower().split()
    embeddings = []
    for word in words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])
    if embeddings:
        sentence_embedding = np.mean(embeddings, axis=0)
    else:
        sentence_embedding = np.zeros(word_vectors.vector_size)
    return sentence_embedding

def compute_bleu(references, hypotheses):
    """
    Compute BLEU-1 to BLEU-4 scores.

    Args:
        references (list of str): Reference captions.
        hypotheses (list of str): Hypothesis captions.

    Returns:
        dict: BLEU-1 to BLEU-4 scores.
    """
    smoothie = SmoothingFunction().method4
    ref_list = [[ref.split()] for ref in references]
    hyp_list = [hyp.split() for hyp in hypotheses]
    bleu_scores = {
        'BLEU-1': corpus_bleu(ref_list, hyp_list, weights=(1, 0, 0, 0), smoothing_function=smoothie),
        'BLEU-2': corpus_bleu(ref_list, hyp_list, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        'BLEU-3': corpus_bleu(ref_list, hyp_list, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie),
        'BLEU-4': corpus_bleu(ref_list, hyp_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    }
    return bleu_scores

def compute_semantic_similarity(references, hypotheses):
    """
    Compute average semantic similarity between references and hypotheses.

    Args:
        references (list of str): Reference captions.
        hypotheses (list of str): Hypothesis captions.

    Returns:
        float: Average cosine similarity.
    """
    similarities = []
    for ref, hyp in zip(references, hypotheses):
        ref_emb = get_sentence_embedding(ref).reshape(1, -1)
        hyp_emb = get_sentence_embedding(hyp).reshape(1, -1)
        similarity = cosine_similarity(ref_emb, hyp_emb)[0][0]
        similarities.append(similarity)
    avg_similarity = np.mean(similarities)
    return avg_similarity

def compute_metrics(references, hypotheses):
    """
    Compute BLEU, CIDEr, METEOR, ROUGE scores, and semantic similarity.

    Args:
        references (list of str): Reference captions.
        hypotheses (list of str): Hypothesis captions.

    Returns:
        tuple: BLEU scores dict, CIDEr score, METEOR score, ROUGE score, Semantic Similarity.
    """
    # BLEU Scores
    bleu_scores = compute_bleu(references, hypotheses)

    # Prepare data for COCO evaluation
    ref_dict = defaultdict(list)
    for idx, ref in enumerate(references):
        ref_dict[idx].append({'image_id': idx, 'caption': ref})

    hyp_dict = {idx: [{'image_id': idx, 'caption': hyp}] for idx, hyp in enumerate(hypotheses)}

    # CIDEr Score
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(ref_dict, hyp_dict)

    # METEOR Score
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(ref_dict, hyp_dict)

    # ROUGE Score
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(ref_dict, hyp_dict)

    # Semantic Similarity
    semantic_similarity = compute_semantic_similarity(references, hypotheses)

    return bleu_scores, cider_score, meteor_score, rouge_score, semantic_similarity
