""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
from typing import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class CustomUNet(nn.Module):
    def __init__(
            self,
            num_channels: int = 1,
            num_classes: int = 1,
            activation: str = 'relu',
            use_batch_norm: bool = True,
            dropout: float = .3,
            dropout_change_per_layer: float = .3,
            dropout_type: str = 'spatial',
            use_attention: bool = False,
            filters: int = 16,
            num_layers: int = 4,
            output_activation: str = 'sigmoid',
            bilinear: bool = False,
    ):
        super(CustomUNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.num_layers = num_layers
        self.filters = filters

        self.inc = DoubleConv(self.num_channels, self.filters)
        self.outc = OutConv(self.filters, self.num_classes)

        self.down_list = []
        self.up_list = []

        factor = 2
        in_channels = self.filters
        for i in range(self.num_layers):
            self.down_list.append(Down(in_channels, in_channels * factor))
            self.up_list.append(Up(in_channels * factor, in_channels))
            in_channels *= factor
        self.up_list = self.up_list[::-1]

    def forward(self, x):
        x = self.inc(x)
        for down in self.down_list:
            x = down(x)
        for up in self.up_list:
            x = up(x)

        logits = self.outc(x)
        return logits
