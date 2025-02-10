# models/feature_extractor.py

import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', fine_tune=False, device='cuda'):
        """
        Feature extractor using a pretrained CNN model.

        Args:
            model_name (str): Name of the pretrained model ('resnet50').
            fine_tune (bool): Whether to fine-tune the CNN layers.
            device (str): Device to run the model on.
        """
        super(FeatureExtractor, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = self.model.fc.in_features
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if fine_tune:
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model = self.model.to(self.device)
        self.model.eval()

    def forward(self, x):
        """
        Extract features from images.

        Args:
            x (Tensor): Batch of images, shape (batch_size, 3, 224, 224).

        Returns:
            Tensor: Extracted features, shape (batch_size, feature_dim).
        """
        with torch.no_grad():
            features = self.model(x)
        return features.view(features.size(0), -1)  # Shape: (batch_size, feature_dim)
