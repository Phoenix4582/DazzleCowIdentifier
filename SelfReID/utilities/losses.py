import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(torch.nn.Module):
    """NTXentLoss (Normalized Temperature-scaled Cross-Entropy Loss) for explicit positive and negative embeddings."""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_pos, z_neg):
        """
        Computes NTXentLoss using explicitly defined positives and negatives.

        Args:
            z_pos (torch.Tensor): Positive pairs embeddings (B, D)
            z_neg (torch.Tensor): Negative pairs embeddings (B, D)

        Returns:
            torch.Tensor: NTXent loss value
        """
        batch_size = z_pos.shape[0]

        # Normalize embeddings (ensures stable cosine similarity values)
        z_pos = F.normalize(z_pos, dim=1)
        z_neg = F.normalize(z_neg, dim=1)

        # Compute cosine similarity between positive pairs (should be high)
        pos_sim = torch.sum(z_pos * z_pos, dim=1)  # Diagonal elements
        pos_sim = pos_sim.unsqueeze(1)  # Shape (B, 1)

        # Compute cosine similarity between negative pairs (should be low)
        neg_sim = torch.mm(z_pos, z_neg.T)  # Shape (B, B), all negatives in batch

        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # Shape (B, B+1)

        # Temperature scaling
        logits /= self.temperature

        # Create labels (positives should be first index in each row)
        labels = torch.zeros(batch_size, dtype=torch.long, device=z_pos.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        return loss

# Triplet Loss (With configurable dist_func, default None stands for Pairwise distance)
def triplet_loss(z1, z2, z3, distance_function=None, margin=0.5):
    z1, z2, z3 = nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1), nn.functional.normalize(z3, dim=1)
    return torch.nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=margin)(z1, z2, z3)

def cosine_distance(x1, x2):
    return 1 - torch.nn.functional.cosine_similarity(x1, x2, dim=-1)
