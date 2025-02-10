"""
LinkNet34.py - Define the neural network for LinkNet34.
Reference - https://ieeexplore.ieee.org/abstract/document/8305148
"""

from torch import nn
from torchvision import models


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(decoder_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(
                in_channels // 4,
                in_channels // 4,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class LinkNet34(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(LinkNet34, self).__init__()
        resnet = models.resnet34(weights=None)
        # resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        self.input_block = nn.Sequential(*list(resnet.children())[0:4])

        self.encoder1 = nn.Sequential(*list(resnet.children())[4])
        self.encoder2 = nn.Sequential(*list(resnet.children())[5])
        self.encoder3 = nn.Sequential(*list(resnet.children())[6])
        self.encoder4 = nn.Sequential(*list(resnet.children())[7])

        self.decoder4 = decoder_block(512, 256, dropout_rate)
        self.decoder3 = decoder_block(256, 128, dropout_rate)
        self.decoder2 = decoder_block(128, 64, dropout_rate)
        self.decoder1 = decoder_block(64, 64, dropout_rate)

        self.output_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 1, kernel_size=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        i1 = self.input_block(x)

        e1 = self.encoder1(i1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        o1 = self.output_block(d1)

        return o1
