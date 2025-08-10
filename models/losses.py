import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class GANLoss(nn.Module):
    """GAN loss for adversarial training.
    
    Supports both LSGAN (least squares) and vanilla GAN losses.
    LSGAN typically provides more stable training.
    """
    
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
        
        self.use_lsgan = use_lsgan

    def get_target_tensor(self, prediction, target_is_real):
        """Create target tensor filled with real/fake labels."""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """Calculate GAN loss.
        
        Args:
            prediction: Discriminator predictions
            target_is_real: Whether the target should be real (True) or fake (False)
            
        Returns:
            GAN loss value
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features.
    
    Compares high-level features from a pre-trained VGG network
    to encourage perceptually similar outputs.
    """
    
    def __init__(self, layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1']):
        super(PerceptualLoss, self).__init__()
        
        # Load pre-trained VGG19
        vgg = vgg19(pretrained=True).features
        self.layers = layers
        self.features = nn.ModuleDict()
        
        # Extract specified layers
        layer_names = {
            '0': 'relu_1_1', '5': 'relu_2_1', '10': 'relu_3_1',
            '19': 'relu_4_1', '28': 'relu_5_1'
        }
        
        for name, module in vgg.named_children():
            if name in layer_names:
                self.features[layer_names[name]] = module
            if layer_names.get(name) == layers[-1]:
                break
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        """Calculate perceptual loss.
        
        Args:
            input: Generated image [B, 1, H, W]
            target: Target image [B, 1, H, W]
            
        Returns:
            Perceptual loss value
        """
        # Convert grayscale to RGB for VGG
        if input.size(1) == 1:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        loss = 0
        x_input = input
        x_target = target
        
        for name, layer in self.features.items():
            x_input = layer(x_input)
            x_target = layer(x_target)
            
            if name in self.layers:
                loss += self.criterion(x_input, x_target)
        
        return loss

class StyleLoss(nn.Module):
    """Style loss using Gram matrices.
    
    Captures texture and style information by comparing
    correlations between feature maps.
    """
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def gram_matrix(self, features):
        """Calculate Gram matrix for style representation."""
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (channels * height * width)

    def forward(self, input, target):
        """Calculate style loss.
        
        Args:
            input: Generated image features
            target: Target image features
            
        Returns:
            Style loss value
        """
        input_gram = self.gram_matrix(input)
        target_gram = self.gram_matrix(target)
        return self.criterion(input_gram, target_gram)

class CombinedLoss(nn.Module):
    """Combined loss function for Pix2Pix training.
    
    Combines adversarial, L1, and optional perceptual losses
    with configurable weights.
    """
    
    def __init__(self, lambda_l1=100, lambda_perceptual=0.1, use_perceptual=False):
        super(CombinedLoss, self).__init__()
        
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual
        
        # Loss functions
        self.gan_loss = GANLoss()
        self.l1_loss = nn.L1Loss()
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()

    def forward(self, fake_pred, fake_image, real_image, target_is_real=True):
        """Calculate combined loss.
        
        Args:
            fake_pred: Discriminator prediction on fake image
            fake_image: Generated image
            real_image: Target real image
            target_is_real: Whether target should be real for GAN loss
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Adversarial loss
        losses['gan'] = self.gan_loss(fake_pred, target_is_real)
        
        # L1 reconstruction loss
        losses['l1'] = self.l1_loss(fake_image, real_image)
        
        # Total loss
        total_loss = losses['gan'] + self.lambda_l1 * losses['l1']
        
        # Optional perceptual loss
        if self.use_perceptual:
            losses['perceptual'] = self.perceptual_loss(fake_image, real_image)
            total_loss += self.lambda_perceptual * losses['perceptual']
        
        losses['total'] = total_loss
        
        return losses

class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for improved training stability.
    
    Matches intermediate features from the discriminator
    instead of just the final output.
    """
    
    def __init__(self, num_layers=3):
        super(FeatureMatchingLoss, self).__init__()
        self.num_layers = num_layers
        self.criterion = nn.L1Loss()

    def forward(self, real_features, fake_features):
        """Calculate feature matching loss.
        
        Args:
            real_features: List of discriminator features for real images
            fake_features: List of discriminator features for fake images
            
        Returns:
            Feature matching loss value
        """
        loss = 0
        for i in range(min(len(real_features), self.num_layers)):
            loss += self.criterion(fake_features[i], real_features[i].detach())
        
        return loss / min(len(real_features), self.num_layers)