# utils/tokenizer.py

import nltk
from collections import Counter
import pickle
import os

class Tokenizer:
    def __init__(self, captions_list=None, vocab_size=5000):
        """
        Tokenizer for encoding and decoding captions.

        Args:
            captions_list (list, optional): List of all captions to build the vocabulary.
            vocab_size (int): Maximum size of the vocabulary.
        """
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        if captions_list is not None:
            self.build_vocab(captions_list)

    def build_vocab(self, captions_list):
        counter = Counter()
        for caption in captions_list:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

        # Most common words
        vocab = counter.most_common(self.vocab_size - 4)  # Reserving 4 for special tokens
        self.word2idx = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }
        idx = 4
        for word, _ in vocab:
            self.word2idx[word] = idx
            idx += 1
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, caption, max_length=20):
        """
        Encode a caption into a list of indices.

        Args:
            caption (str): The caption to encode.
            max_length (int): Maximum length of the encoded caption.

        Returns:
            list: List of encoded indices.
        """
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokens = ['<SOS>'] + tokens + ['<EOS>']
        if len(tokens) < max_length:
            tokens += ['<PAD>'] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            if tokens[-1] != '<EOS>':
                tokens[-1] = '<EOS>'
        return [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]

    def decode(self, indices):
        """
        Decode a list of indices back into a caption.

        Args:
            indices (list or ndarray): List of indices to decode.

        Returns:
            str: The decoded caption.
        """
        tokens = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        caption = []
        for token in tokens:
            if token == '<EOS>':
                break
            if token not in ('<SOS>', '<PAD>'):
                caption.append(token)
        return ' '.join(caption)

    def save(self, filepath):
        """
        Save the tokenizer to a file.

        Args:
            filepath (str): Path to save the tokenizer.
        """
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Tokenizer saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a tokenizer from a file.

        Args:
            filepath (str): Path to the tokenizer file.

        Returns:
            Tokenizer: Loaded tokenizer instance.
        """
        with open(filepath, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"Tokenizer loaded from {filepath}")
        return tokenizer
