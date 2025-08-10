import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """PatchGAN Discriminator for Pix2Pix.
    
    Classifies 70x70 patches as real or fake instead of the entire image.
    This encourages the generator to produce high-frequency details.
    """
    
    def __init__(self, in_channels=2):  # sketch + photo channels
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Create a discriminator block with conv, batchnorm, and leaky relu."""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Build discriminator architecture
        # Input: 256x256 -> Output: 30x30 patches
        self.model = nn.Sequential(
            # Layer 1: 256 -> 128 (no batchnorm for first layer)
            *discriminator_block(in_channels, 64, normalization=False),
            
            # Layer 2: 128 -> 64
            *discriminator_block(64, 128),
            
            # Layer 3: 64 -> 32
            *discriminator_block(128, 256),
            
            # Layer 4: 32 -> 16
            *discriminator_block(256, 512),
            
            # Final layer: 16 -> 15 (patch classification)
            nn.ZeroPad2d((1, 0, 1, 0)),  # Asymmetric padding
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        """Forward pass through discriminator.
        
        Args:
            img_A: Input sketch tensor [B, 1, 256, 256]
            img_B: Target/generated photo tensor [B, 1, 256, 256]
            
        Returns:
            Patch predictions tensor [B, 1, 30, 30]
        """
        # Concatenate sketch and photo as input
        img_input = torch.cat((img_A, img_B), 1)  # [B, 2, 256, 256]
        
        # Pass through discriminator network
        output = self.model(img_input)  # [B, 1, 30, 30]
        
        return output

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for improved training stability.
    
    Uses multiple discriminators at different image scales to capture
    both global structure and local details.
    """
    
    def __init__(self, in_channels=2, num_discriminators=2):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.num_discriminators = num_discriminators
        
        # Create multiple discriminators
        self.discriminators = nn.ModuleList()
        for i in range(num_discriminators):
            self.discriminators.append(Discriminator(in_channels))
        
        # Downsampling for multi-scale inputs
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, img_A, img_B):
        """Forward pass through multi-scale discriminator.
        
        Args:
            img_A: Input sketch tensor [B, 1, 256, 256]
            img_B: Target/generated photo tensor [B, 1, 256, 256]
            
        Returns:
            List of patch prediction tensors at different scales
        """
        results = []
        
        input_A, input_B = img_A, img_B
        
        for i in range(self.num_discriminators):
            # Apply discriminator at current scale
            output = self.discriminators[i](input_A, input_B)
            results.append(output)
            
            # Downsample for next scale (except for last discriminator)
            if i < self.num_discriminators - 1:
                input_A = self.downsample(input_A)
                input_B = self.downsample(input_B)
        
        return results