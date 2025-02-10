# models/caption_generator.py

import torch
import torch.nn as nn
from models.attention import BahdanauAttention
from models.feature_extractor import FeatureExtractor
import torch.utils.checkpoint as checkpoint  # Added missing import

class CaptionGenerator(nn.Module):
    def __init__(
        self,
        embed_dim,
        encoder_dim,
        decoder_dim,
        vocab_size,
        attention_dim,
        device='cuda',
        rnn_type='LSTM'
    ):
        """
        Image Caption Generator with Attention Mechanism.

        Args:
            embed_dim (int): Embedding dimension.
            encoder_dim (int): Feature size of encoder.
            decoder_dim (int): Size of decoder's RNN.
            vocab_size (int): Vocabulary size.
            attention_dim (int): Dimension of attention layer.
            device (str): Device to run the model on.
            rnn_type (str): Type of RNN to use ('LSTM' or 'GRU').
        """
        super(CaptionGenerator, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.5)
        self.rnn_type = rnn_type.upper()

        if self.rnn_type == 'LSTM':
            self.decoder = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        elif self.rnn_type == 'GRU':
            self.decoder = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
        else:
            raise ValueError("rnn_type must be either 'LSTM' or 'GRU'")

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim) if self.rnn_type == 'LSTM' else None
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # Initialize Feature Extractor
        self.feature_extractor = FeatureExtractor(model_name='resnet50', fine_tune=False, device=device)
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

    def forward(self, features, captions, precomputed=False):
        """
        Forward pass for training.

        Args:
            features (Tensor): Extracted features or images, shape (batch_size, encoder_dim) or (batch_size, 3, 224, 224).
            captions (Tensor): Batch of encoded captions, shape (batch_size, max_caption_length).
            precomputed (bool): Flag indicating if features are precomputed.

        Returns:
            Tensor: Output predictions, shape (batch_size, max_caption_length, vocab_size).
        """
        if not precomputed:
            features = self.extract_features(features)  # Shape: (batch_size, encoder_dim)

        batch_size = features.size(0)
        encoder_outputs = features  # Shape: (batch_size, encoder_dim)

        # Initialize RNN state
        mean_encoder_output = encoder_outputs.mean(dim=1)
        h = self.init_h(mean_encoder_output).to(self.device)
        c = self.init_c(mean_encoder_output).to(self.device) if self.rnn_type == 'LSTM' else None

        embeddings = self.embedding(captions).to(self.device)  # Shape: (batch_size, max_caption_length, embed_dim)
        max_caption_length = captions.size(1)
        outputs = torch.zeros(batch_size, max_caption_length, self.fc.out_features).to(self.device)

        for t in range(max_caption_length):
            context, alpha = checkpoint.checkpoint(self.attention, encoder_outputs, h)
            input_combined = torch.cat([embeddings[:, t, :], context], dim=1)  # Shape: (batch_size, embed_dim + encoder_dim)
            if self.rnn_type == 'LSTM':
                h, c = self.decoder(input_combined, (h, c))  # h, c: (batch_size, decoder_dim)
            else:
                h = self.decoder(input_combined, h)  # h: (batch_size, decoder_dim)

            # Apply f_beta gating
            attention_scale = torch.sigmoid(self.f_beta(h))  # Shape: (batch_size, encoder_dim)
            context = attention_scale * context  # Shape: (batch_size, encoder_dim)

            output = self.fc(self.dropout(h))  # Shape: (batch_size, vocab_size)
            outputs[:, t, :] = output

        return outputs

    def generate(self, features, tokenizer, max_length=20, precomputed=False):
        """
        Generate captions for given images or precomputed features.

        Args:
            features (Tensor): Extracted features or images, shape (batch_size, encoder_dim) or (batch_size, 3, 224, 224).
            tokenizer (Tokenizer): Tokenizer instance.
            max_length (int): Maximum length of the generated caption.
            precomputed (bool): Flag indicating if features are precomputed.

        Returns:
            tuple: Generated captions tensor (batch_size, max_length) and attention weights list.
        """
        self.eval()
        with torch.no_grad():
            if not precomputed:
                features = self.extract_features(features)
            batch_size = features.size(0)
            encoder_outputs = features  # Shape: (batch_size, encoder_dim)

            # Initialize RNN state
            mean_encoder_output = encoder_outputs.mean(dim=1)
            h = self.init_h(mean_encoder_output).to(self.device)
            c = self.init_c(mean_encoder_output).to(self.device) if self.rnn_type == 'LSTM' else None

            # Initialize input with <SOS> tokens
            inputs = torch.tensor([tokenizer.word2idx['<SOS>']] * batch_size).to(self.device)
            embeddings = self.embedding(inputs).to(self.device)  # Shape: (batch_size, embed_dim)
            outputs = []
            alphas = []

            for _ in range(max_length):
                context, alpha = checkpoint.checkpoint(self.attention, encoder_outputs, h)  # context: (batch_size, encoder_dim)
                input_combined = torch.cat([embeddings, context], dim=1)  # Shape: (batch_size, embed_dim + encoder_dim)

                if self.rnn_type == 'LSTM':
                    h, c = self.decoder(input_combined, (h, c))  # h, c: (batch_size, decoder_dim)
                else:
                    h = self.decoder(input_combined, h)  # h: (batch_size, decoder_dim)

                # Apply f_beta gating
                attention_scale = torch.sigmoid(self.f_beta(h))  # Shape: (batch_size, encoder_dim)
                context = attention_scale * context  # Shape: (batch_size, encoder_dim)

                output = self.fc(self.dropout(h))  # Shape: (batch_size, vocab_size)
                predicted = output.argmax(1)  # Shape: (batch_size,)

                outputs.append(predicted)
                alphas.append(alpha)

                embeddings = self.embedding(predicted).to(self.device)  # Prepare embedding for next time step

            outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, max_length)
            return outputs, alphas

    def extract_features(self, images):
        """
        Extract features from images using the FeatureExtractor.

        Args:
            images (Tensor): Batch of images, shape (batch_size, 3, 224, 224).

        Returns:
            Tensor: Extracted features, shape (batch_size, encoder_dim).
        """
        with torch.no_grad():
            features = self.feature_extractor(images)
        return features
