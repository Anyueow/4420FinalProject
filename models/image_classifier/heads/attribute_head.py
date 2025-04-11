"""
Custom attribute head for fashion classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttributeHead(nn.Module):
    """
    Custom head for fashion attribute classification.
    Predicts multiple attributes (color, pattern, style) from RoI features.
    """
    
    def __init__(self, 
                 in_channels,
                 num_attributes,
                 hidden_dim=1024):
        super().__init__()
        
        self.attribute_predictor = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_attributes)
        )
        
    def forward(self, x):
        """
        Forward pass of the attribute head.
        Args:
            x: RoI features from the backbone
        Returns:
            Attribute predictions
        """
        # Flatten if needed
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
            
        return self.attribute_predictor(x)
    
class FocalLoss(nn.Module):
    """
    Focal Loss implementation for attribute classification.
    Helps handle class imbalance in attribute predictions.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions, targets):
        """
        Calculate focal loss.
        Args:
            predictions: Model predictions
            targets: Ground truth labels
        Returns:
            Computed focal loss
        """
        ce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        return focal_loss.mean()

def build_attribute_head(config):
    """Build attribute head with config."""
    head = AttributeHead(
        in_channels=config.get('in_channels', 256),
        num_attributes=config.get('num_attributes', 294),  # iMaterialist attributes
        hidden_dim=config.get('hidden_dim', 1024)
    )
    return head 