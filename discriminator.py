import torch
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 256, 2),
            ConvBlock(256, 512, 1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.model(x) 
    
    
