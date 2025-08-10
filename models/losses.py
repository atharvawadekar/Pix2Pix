import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class GANLoss(nn.Module):
    """GAN loss for gen vs dis
    
    both least squares and vanilla gan losses
    
    LSGAN more stable
    
    i guess have to see which one is better for us and keep that next time we go through this
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
        """target tensor filled with labels"""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """Calculate GAN loss.
        
        Args:
            prediction: discriminator predictions
            target_is_real: ground truth
            
        Returns:
            GAN loss value
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features.
    
    compares high level features from pretrained VGG network
    to encourage perceptually similar outputs.
    """
    
    def __init__(self, layers=['relu_1_1', 'relu_2_1', 'relu_3_1', 'relu_4_1', 'relu_5_1']):
        super(PerceptualLoss, self).__init__()
        
        # load pretrained VGG19
        vgg = vgg19(pretrained=True).features
        self.layers = layers
        self.features = nn.ModuleDict()
        
        # extract specified layers
        layer_names = {
            '0': 'relu_1_1', '5': 'relu_2_1', '10': 'relu_3_1',
            '19': 'relu_4_1', '28': 'relu_5_1'
        }
        
        for name, module in vgg.named_children():
            if name in layer_names:
                self.features[layer_names[name]] = module
            if layer_names.get(name) == layers[-1]:
                break
        
        # freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()

    def forward(self, input, target):
        """calculate perceptual loss.
        
        Args:
            input: generated image [B, 1, H, W]
            target: target image [B, 1, H, W]
            
        Returns:
            perceptual loss value
        """
        # convert grayscale to RGB for VGG
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
    """style loss using gram matrices.
    
    captiures texture and style by feature map comparison
    """
    
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def gram_matrix(self, features):
        """gram matrix for style representation."""
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (channels * height * width)

    def forward(self, input, target):
        """calculate style loss.
        
        Args:
            input: generated image features
            target: target image features
            
        Returns:
            style loss value
        """
        input_gram = self.gram_matrix(input)
        target_gram = self.gram_matrix(target)
        return self.criterion(input_gram, target_gram)

class CombinedLoss(nn.Module):
    """pix2pix combined loss funtion
    
    gan + L1 + perceptual (have to see if perceptual actually used)
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
        """calculate combined loss.
        
        Args:
            fake_pred: discriminator prediction on fake image
            fake_image: generated image
            real_image: target real image
            target_is_real: ground truth
            
        Returns:
            dict with individual and total losses
        """
        losses = {}
        
        # gan
        losses['gan'] = self.gan_loss(fake_pred, target_is_real)
        
        # L1 
        losses['l1'] = self.l1_loss(fake_image, real_image)
        
        # total loss
        total_loss = losses['gan'] + self.lambda_l1 * losses['l1']
        
        # optional perceptual loss
        if self.use_perceptual:
            losses['perceptual'] = self.perceptual_loss(fake_image, real_image)
            total_loss += self.lambda_perceptual * losses['perceptual']
        
        losses['total'] = total_loss
        
        return losses

class FeatureMatchingLoss(nn.Module):
    """feature matching loss, stable++
    
    intermediate features lso matched along with final output
    """
    
    def __init__(self, num_layers=3):
        super(FeatureMatchingLoss, self).__init__()
        self.num_layers = num_layers
        self.criterion = nn.L1Loss()

    def forward(self, real_features, fake_features):
        """calculate feature matching loss.
        
        Args:
            real_features: list of discriminator features for real images
            fake_features: list of discriminator features for fake images
            
        Returns:
            feature matching loss value
        """
        loss = 0
        for i in range(min(len(real_features), self.num_layers)):
            loss += self.criterion(fake_features[i], real_features[i].detach())
        
        return loss / min(len(real_features), self.num_layers)