"""Custom models.

Authors
 * Pooneh Mousavi 2024
"""

# Adapted from:
# https://github.com/poonehmousavi/speechbrain/blob/8a11d5d6cf159838996788e10a674cba9bc8e87e/recipes/VoxCeleb/SpeakerRec/discrete/custom_model.py

import torch


__all__ = ["AttentionMLP"]


class AttentionMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x):
        x = self.layers(x)
        att_w = torch.nn.functional.softmax(x, dim=2)
        return att_w
