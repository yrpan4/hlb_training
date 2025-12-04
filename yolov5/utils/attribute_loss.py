# Ultralytics 🚀 AGPL-3.0 License
"""Loss function for multi-task attribute learning."""

import torch
import torch.nn as nn


class ComputeAttributeLoss:
    """Computes attribute (symmetry and vein_color) losses for multi-task learning."""

    def __init__(self, attr_weight=2.0, device='cpu'):
        """Initialize attribute loss computer.
        
        Args:
            attr_weight: weight for attribute losses relative to detection loss (default 2.0)
            device: torch device
        """
        self.criterion = nn.CrossEntropyLoss()
        self.attr_weight = attr_weight
        self.device = device

    def __call__(self, sym_logits, vein_logits, attr_targets):
        """Compute attribute losses.
        
        Args:
            sym_logits: (B, 2) logits for symmetry prediction
            vein_logits: (B, 2) logits for vein_color prediction
            attr_targets: (B, 4) tensor with [sym_mask, sym_label, vein_mask, vein_label]
                where mask is 0/1 (1 = labeled, 0 = unlabeled), label is 0/1
        
        Returns:
            total_attr_loss: weighted sum of symmetry and vein_color losses
            loss_dict: dict with individual loss values
        """
        sym_mask = attr_targets[:, 0].bool()
        sym_label = attr_targets[:, 1].long()
        vein_mask = attr_targets[:, 2].bool()
        vein_label = attr_targets[:, 3].long()

        # Only compute loss for labeled samples
        sym_loss = torch.tensor(0.0, device=self.device)
        vein_loss = torch.tensor(0.0, device=self.device)

        if sym_mask.sum() > 0:
            sym_loss = self.criterion(sym_logits[sym_mask], sym_label[sym_mask])

        if vein_mask.sum() > 0:
            vein_loss = self.criterion(vein_logits[vein_mask], vein_label[vein_mask])

        total_attr_loss = (sym_loss + vein_loss) * self.attr_weight

        return total_attr_loss, {
            "sym_loss": sym_loss.item(),
            "vein_loss": vein_loss.item(),
            "total_attr_loss": total_attr_loss.item(),
        }
