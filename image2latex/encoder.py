import torch
from torch import nn, Tensor


class Encoder(nn.Module):
    def __init__(self, enc_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            nn.Conv2d(64, 128, 3, 1),
            nn.Conv2d(128, 256, 3, 1),
            nn.Conv2d(256, 256, 3, 1),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(256, 512, 3, 1),
            nn.MaxPool2d(1, 2),
            nn.Conv2d(512, enc_dim, 3, 1),
        )
        self.conv.apply(self.init_weights)

    def init_weights(self, layer: nn.Module):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x: Tensor):
        """
            x: (bs, c, w, h)
        """
        return self.conv(x)
