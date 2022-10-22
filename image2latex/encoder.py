import torch
from torch import nn, Tensor
import torchvision


class ConvWithRowEncoder(nn.Module):
    def __init__(self, enc_dim: int):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(1, 2),
            nn.Conv2d(512, 512, 3, 1),
            nn.BatchNorm2d(512),
        )

        self.row_encoder = nn.LSTM(512, enc_dim, batch_first=True, bidirectional=True)

        self.enc_dim = enc_dim * 2  # bidirectional = True

    def forward(self, x: Tensor):
        """
            x: (bs, c, w, h)
        """
        conv_out = self.feature_encoder(x)  # (bs, c, w, h)
        conv_out = conv_out.permute(0, 2, 3, 1)  # (bs, w, h, c)

        bs, w, h, c = conv_out.size()
        rnn_out = []
        for row in range(w):
            row_data = conv_out[:, row, :, :]  # take a row data
            row_out, (h, c) = self.row_encoder(row_data)
            rnn_out.append(row_out)

        encoder_out = torch.stack(rnn_out, dim=1)
        bs, _, _, d = encoder_out.size()
        encoder_out = encoder_out.view(bs, -1, d)

        return encoder_out


class ConvEncoder(nn.Module):
    def __init__(self, enc_dim: int):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            nn.Conv2d(64, 128, 3, 1),
            nn.Conv2d(128, 256, 3, 1),
            nn.Conv2d(256, 256, 3, 1),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(256, 512, 3, 1),
            nn.MaxPool2d(1, 2),
            nn.Conv2d(512, enc_dim, 3, 1),
        )
        self.enc_dim = enc_dim

    def forward(self, x: Tensor):
        """
            x: (bs, c, w, h)
        """
        encoder_out = self.feature_encoder(x)  # (bs, c, w, h)
        encoder_out = encoder_out.permute(0, 2, 3, 1)  # (bs, w, h, c)
        bs, _, _, d = encoder_out.size()
        encoder_out = encoder_out.view(bs, -1, d)
        return encoder_out


class ConvBNEncoder(nn.Module):
    def __init__(self, enc_dim: int):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(1, 2),
            nn.Conv2d(512, enc_dim, 3, 1),
            nn.BatchNorm2d(enc_dim),
        )
        self.enc_dim = enc_dim

    def forward(self, x: Tensor):
        """
            x: (bs, c, w, h)
        """
        encoder_out = self.feature_encoder(x)  # (bs, c, w, h)
        encoder_out = encoder_out.permute(0, 2, 3, 1)  # (bs, w, h, c)
        bs, _, _, d = encoder_out.size()
        encoder_out = encoder_out.view(bs, -1, d)
        return encoder_out


class ResNetEncoder(nn.Module):
    def __init__(self, enc_dim: int):
        super().__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(512, enc_dim)
        self.enc_dim = enc_dim

    def forward(self, x: Tensor):
        """
            x: (bs, c, w, h)
        """
        out = self.resnet(x)
        out = out.permute(0, 2, 3, 1)
        out = self.fc(out)
        bs, _, _, d = out.size()
        out = out.view(bs, -1, d)
        return out


class ResNetWithRowEncoder(nn.Module):
    def __init__(self, enc_dim: int):
        super().__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.row_encoder = nn.LSTM(512, enc_dim, batch_first=True, bidirectional=True)

        self.enc_dim = enc_dim * 2  # bidirectional = True

    def forward(self, x: Tensor):
        """
            x: (bs, c, w, h)
        """
        conv_out = self.resnet(x)
        conv_out = conv_out.permute(0, 2, 3, 1)
        bs, w, h, c = conv_out.size()
        rnn_out = []
        for row in range(w):
            row_data = conv_out[:, row, :, :]  # take a row data
            row_out, (h, c) = self.row_encoder(row_data)
            rnn_out.append(row_out)

        encoder_out = torch.stack(rnn_out, dim=1)
        bs, _, _, d = encoder_out.size()
        encoder_out = encoder_out.view(bs, -1, d)
        return encoder_out
