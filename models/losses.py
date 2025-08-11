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

