import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """PatchGAN discriminator
    
    each 70x70 patch labeled as either real or fake. Detailed images created by generator.
    """
    
    def __init__(self, in_channels=2):  # sketch & photo channels
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """discriminator block with conv, batchnorm and leaky rule."""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # build discriminator architecture
        # 256x256 to 15x15
        self.model = nn.Sequential(
            # 256 to 128, without batch norm
            *discriminator_block(in_channels, 64, normalization=False),
            
            # 128 to 64
            *discriminator_block(64, 128),
            
            # 64 to 32
            *discriminator_block(128, 256),
            
            # 32 to 16
            *discriminator_block(256, 512),
            
            # 16 to 15 along with patch labelling
            nn.ZeroPad2d((1, 0, 1, 0)),  # asymmetric padding
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        """forward pass through discriminator
        
        args:
            img_A: input sketch tensor [B, 1, 256, 256]
            img_B: target photo tensor [B, 1, 256, 256]
            
        returns:
            patch predictions tensor [B, 1, 30, 30]
        """
        # sketch and photo concatenated as input
        img_input = torch.cat((img_A, img_B), 1)  # [B, 2, 256, 256]
        
        # pass through discriminator network
        output = self.model(img_input)  # [B, 1, 15, 15]
        
        return output