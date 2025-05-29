import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class YoloTiny(nn.Module):
    def __init__(self, num_classes=20, num_bins=16):
        super().__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.output_channels = 4 * num_bins + 1 + num_classes  # DFL bbox + objectness + class logits

        self.backbone = nn.Sequential(
            ConvBlock(3, 16, 3, 1, 1),  # [B, 16, 448, 448]
            nn.MaxPool2d(2, 2),         # [B, 16, 224, 224]
            ConvBlock(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),         # [B, 32, 112, 112]
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),         # [B, 64, 56, 56]
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),         # [B, 128, 28, 28]
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),         # [B, 256, 14, 14]
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),         # [B, 512, 7, 7]
        )

        self.head = nn.Sequential(
            ConvBlock(512, 256, 3, 1, 1),
            nn.Conv2d(256, self.output_channels, kernel_size=1)  # [B, output_channels, 7, 7]
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)                  # [B, output_channels, 7, 7]
        x = x.permute(0, 2, 3, 1)         # ➜ [B, 7, 7, output_channels]
        return x
