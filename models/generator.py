import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    """U-Net building block with skip connections."""
    
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)
        self.down = down
        self.use_dropout = use_dropout
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.down:
            out = self.conv(x)
            out = self.bn(out)
            out = self.leaky(out)
        else:
            out = self.conv_t(x)
            out = self.bn(out)
            if self.use_dropout:
                out = self.dropout(out)
            out = self.relu(out)
        return out

class Generator(nn.Module):
    """U-Net Generator for Pix2Pix.
    
    Converts face sketches to realistic photos using encoder-decoder
    architecture with skip connections.
    """
    
    def __init__(self, in_channels=1, out_channels=1):
        super(Generator, self).__init__()
        
        # Encoder (Downsampling path)
        self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)  # 256 -> 128
        self.down2 = UNetBlock(64, 128, down=True)        # 128 -> 64
        self.down3 = UNetBlock(128, 256, down=True)       # 64 -> 32
        self.down4 = UNetBlock(256, 512, down=True)       # 32 -> 16
        self.down5 = UNetBlock(512, 512, down=True)       # 16 -> 8
        self.down6 = UNetBlock(512, 512, down=True)       # 8 -> 4
        self.down7 = UNetBlock(512, 512, down=True)       # 4 -> 2
        self.down8 = nn.Conv2d(512, 512, 4, 2, 1)         # 2 -> 1 (bottleneck)
        
        # Decoder (Upsampling path with skip connections)
        self.up1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)  # 1 -> 2
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)  # 2 -> 4
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)  # 4 -> 8
        self.up4 = UNetBlock(1024, 512, down=False, use_dropout=True)  # 8 -> 16
        self.up5 = UNetBlock(1024, 256, down=False)       # 16 -> 32
        self.up6 = UNetBlock(512, 128, down=False)        # 32 -> 64
        self.up7 = UNetBlock(256, 64, down=False)         # 64 -> 128
        self.up8 = nn.ConvTranspose2d(128, out_channels, 4, 2, 1)  # 128 -> 256
        
        # Activation functions
        self.leaky = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Forward pass through U-Net.
        
        Args:
            x: Input sketch tensor [B, 1, 256, 256]
            
        Returns:
            Generated photo tensor [B, 1, 256, 256]
        """
        # Encoder path - progressively downsample
        d1 = self.leaky(self.down1(x))      # [B, 64, 128, 128]
        d2 = self.down2(d1)                 # [B, 128, 64, 64]
        d3 = self.down3(d2)                 # [B, 256, 32, 32]
        d4 = self.down4(d3)                 # [B, 512, 16, 16]
        d5 = self.down5(d4)                 # [B, 512, 8, 8]
        d6 = self.down6(d5)                 # [B, 512, 4, 4]
        d7 = self.down7(d6)                 # [B, 512, 2, 2]
        d8 = self.leaky(self.down8(d7))     # [B, 512, 1, 1] - bottleneck
        
        # Decoder path - progressively upsample with skip connections
        u1 = self.dropout(self.relu(self.up1(d8)))          # [B, 512, 2, 2]
        u2 = self.up2(torch.cat([u1, d7], 1))               # [B, 512, 4, 4]
        u3 = self.up3(torch.cat([u2, d6], 1))               # [B, 512, 8, 8]
        u4 = self.up4(torch.cat([u3, d5], 1))               # [B, 512, 16, 16]
        u5 = self.up5(torch.cat([u4, d4], 1))               # [B, 256, 32, 32]
        u6 = self.up6(torch.cat([u5, d3], 1))               # [B, 128, 64, 64]
        u7 = self.up7(torch.cat([u6, d2], 1))               # [B, 64, 128, 128]
        u8 = self.tanh(self.up8(torch.cat([u7, d1], 1)))    # [B, 1, 256, 256]
        
        return u8