import torch
import torch.nn as nn
from core.config import NUM_ITEMS

class MenuUpsellModel(nn.Module):
    def __init__(self, num_items=NUM_ITEMS, emb_dim=16):
        super().__init__()
        self.anchor_emb = nn.Embedding(num_items, emb_dim)
        self.target_emb = nn.Embedding(num_items, emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, anchor_ids, target_ids):
        a = self.anchor_emb(anchor_ids)
        t = self.target_emb(target_ids)
        return self.classifier(torch.cat([a, t], dim=1)).squeeze(-1)