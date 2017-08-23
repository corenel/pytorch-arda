"""Classifier for ARDA.

guarantee learned domain-invariant representations are discriminative enough
to accomplish the final classification task
"""

import torch.nn.functional as F
from torch import nn


class Classifier(nn.Module):
    """LeNet classifier model for ARDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
