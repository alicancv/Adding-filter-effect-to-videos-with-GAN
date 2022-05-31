import torch
import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, downward=True, activation="relu", use_dropout=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if downward else nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels),
            nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2)
        )

        self.downward = downward
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        self.down1 = ConvBlock(64, 128, activation="leaky", downward=True, use_dropout=False)
        self.down2 = ConvBlock(128, 256, activation="leaky", downward=True, use_dropout=False)
        self.down3 = ConvBlock(256, 512, activation="leaky", downward=True, use_dropout=False)
        self.down4 = ConvBlock(512, 512, activation="leaky", downward=True, use_dropout=False)
        self.down5 = ConvBlock(512, 512, activation="leaky", downward=True, use_dropout=False)
        self.down6 = ConvBlock(512, 512, activation="leaky", downward=True, use_dropout=False)
        
        self.floor = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), 
            nn.ReLU()
        )

        self.up1 = ConvBlock(512, 512, activation="relu", downward=False, use_dropout=True)
        self.up2 = ConvBlock(1024, 512, activation="relu", downward=False, use_dropout=True)
        self.up3 = ConvBlock(1024, 512, activation="relu", downward=False, use_dropout=True)
        self.up4 = ConvBlock(1024, 512, activation="relu", downward=False, use_dropout=False)
        self.up5 = ConvBlock(1024, 256, activation="relu", downward=False, use_dropout=False)
        self.up6 = ConvBlock(512, 128, activation="relu", downward=False, use_dropout=False)
        self.up7 = ConvBlock(256, 64, activation="relu", downward=False, use_dropout=False)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        down1 = self.initial(x)
        down2 = self.down1(down1)
        down3 = self.down2(down2)
        down4 = self.down3(down3)
        down5 = self.down4(down4)
        down6 = self.down5(down5)
        down7 = self.down6(down6)
        floor = self.floor(down7)
        up1 = self.up1(floor)
        up2 = self.up2(torch.cat([up1, down7], 1))
        up3 = self.up3(torch.cat([up2, down6], 1))
        up4 = self.up4(torch.cat([up3, down5], 1))
        up5 = self.up5(torch.cat([up4, down4], 1))
        up6 = self.up6(torch.cat([up5, down3], 1))
        up7 = self.up7(torch.cat([up6, down2], 1))
        return self.final(torch.cat([up7, down1], 1))


    