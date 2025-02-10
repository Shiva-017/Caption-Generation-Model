# models/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Bahdanau Attention Module.

        Args:
            encoder_dim (int): Feature size of encoded images.
            decoder_dim (int): Size of decoder's RNN.
            attention_dim (int): Size of the attention network.
        """
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Linear layer to transform encoder output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Linear layer to transform decoder hidden state
        self.full_att = nn.Linear(attention_dim, 1)  # Linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass for attention.

        Args:
            encoder_out (Tensor): Encoded images, shape (batch_size, encoder_dim).
            decoder_hidden (Tensor): Previous hidden state of decoder, shape (batch_size, decoder_dim).

        Returns:
            context (Tensor): Weighted sum of encoder features, shape (batch_size, encoder_dim).
            alpha (Tensor): Attention weights, shape (batch_size, 1).
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch_size, 1)
        alpha = self.softmax(att)  # (batch_size, 1)
        context = encoder_out * alpha  # (batch_size, encoder_dim)
        return context, alpha
