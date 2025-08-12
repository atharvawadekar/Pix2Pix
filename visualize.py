import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

def denormalize_tensor(tensor):
    """[-1,1] scaled to [0,1] for visualisation"""
    return (tensor + 1.0) / 2.0

def save_sample_images(generator, dataloader, device, epoch, output_dir, num_samples=8):
    """smaple image with grid containing sketch, generated image and real image in order
    
    Args:
        generator: trained generator model
        dataloader: validation dataloader
        device:  cpu gpu
        epoch: current epoch number
        output_dir: directory to save images
        num_samples: number of samples to visualize
    """
    generator.eval()
    
    with torch.no_grad():
        # validation data batch
        sketches, photos = next(iter(dataloader))
        sketches = sketches.to(device)
        photos = photos.to(device)
        
        # generate images
        fake_photos = generator(sketches)
        
        # select number specified
        sketches = sketches[:num_samples]
        fake_photos = fake_photos[:num_samples]
        photos = photos[:num_samples]
        
        # denormalize images for visualization
        sketches_vis = denormalize_tensor(sketches)
        fake_photos_vis = denormalize_tensor(fake_photos)
        photos_vis = denormalize_tensor(photos)
        
        # create comparison grid: [sketches, generated, real]
        comparison = torch.cat([sketches_vis, fake_photos_vis, photos_vis], dim=0)
        
        # save grid image
        output_path = Path(output_dir) / f"epoch_{epoch:03d}.png"
        vutils.save_image(
            comparison, 
            output_path, 
            nrow=num_samples, 
            normalize=False,
            padding=2,
            pad_value=1.0  # white padding
        )
        
        print(f"Sample images saved to: {output_path}")
