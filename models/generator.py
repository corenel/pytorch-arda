"""Generator for ARDA.

learn the domain-invariant feature representations from inputs across domains.
"""

from torch import nn


class Generator(nn.Module):
    """LeNet encoder model for ARDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(Generator, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv block
            # input [1 x 28 x 28]
            # output [64 x 12 x 12]
            nn.Conv2d(1, 64, 5, 1, 0, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 2nd conv block
            # input [64 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(64, 50, 5, 1, 0, bias=False),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat
