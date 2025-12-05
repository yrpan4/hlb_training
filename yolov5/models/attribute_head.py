# Ultralytics 🚀 AGPL-3.0 License
"""Attribute prediction heads for multi-task learning with YOLOv5."""

import torch
import torch.nn as nn


class AttributeHead(nn.Module):
    """Multi-task attribute prediction head for symmetry and vein_color.
    
    Predicts:
    - symmetry (binary): 0 = symmetric, 1 = asymmetric
    - vein_color (binary): 0 = green, 1 = yellow
    """

    def __init__(self, in_channels=256, hidden_dim=128, dropout=0.5):
        """Initialize attribute head with feature extraction backbone.
        
        Args:
            in_channels: number of input channels from detector backbone (default 256)
            hidden_dim: hidden layer size (default 128)
            dropout: dropout rate (default 0.5)
        """
        super().__init__()
        
        # Global average pooling will be applied before this head
        # Input: (B, in_channels) after GAP
        
        # Projection layer to normalize feature dimension to 256 (in case actual input differs)
        self.proj = nn.Linear(in_channels, 256) if in_channels != 256 else nn.Identity()
        
        # Shared feature extraction
        self.fc1 = nn.Linear(256, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Symmetry head (binary classification)
        self.sym_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # 2 classes: symmetric (0), asymmetric (1)
        )
        
        # Vein color head (binary classification)
        self.vein_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),  # 2 classes: green (0), yellow (1)
        )
        
    def forward(self, x):
        """Forward pass to predict attributes.
        
        Args:
            x: input features of shape (B, C) after global average pooling
        
        Returns:
            sym_logits: (B, 2) logits for symmetry
            vein_logits: (B, 2) logits for vein_color
        """
        # Project input to standard dimension
        x = self.proj(x)
        
        # Shared feature extraction
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Task-specific heads
        sym_logits = self.sym_head(x)
        vein_logits = self.vein_head(x)
        
        return sym_logits, vein_logits
