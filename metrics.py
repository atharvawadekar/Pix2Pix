import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images
    
    Args:
        img1, img2: torch tensors or numpy arrays with values in [0, max_val]
        max_val: maximum possible pixel value
    
    Returns:
        PSNR value in dB
    """
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Ensure images are in range [0, max_val]
    img1 = np.clip(img1, 0, max_val)
    img2 = np.clip(img2, 0, max_val)
    
    return psnr(img1, img2, data_range=max_val)

def calculate_ssim(img1, img2, max_val=1.0):
    """
    Calculate SSIM between two images
    
    Args:
        img1, img2: torch tensors or numpy arrays with values in [0, max_val]
        max_val: maximum possible pixel value
    
    Returns:
        SSIM value between 0 and 1
    """
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Ensure images are in range [0, max_val]
    img1 = np.clip(img1, 0, max_val)
    img2 = np.clip(img2, 0, max_val)
    
    # Handle different input shapes
    if len(img1.shape) == 4:  # Batch of images (B, C, H, W)
        ssim_values = []
        for i in range(img1.shape[0]):
            if img1.shape[1] == 3:  # RGB
                # Convert from (C, H, W) to (H, W, C)
                img1_single = np.transpose(img1[i], (1, 2, 0))
                img2_single = np.transpose(img2[i], (1, 2, 0))
                ssim_val = ssim(img1_single, img2_single, data_range=max_val, multichannel=True, channel_axis=2)
            else:  # Grayscale
                ssim_val = ssim(img1[i, 0], img2[i, 0], data_range=max_val)
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    elif len(img1.shape) == 3:  # Single image (C, H, W)
        if img1.shape[0] == 3:  # RGB
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
            return ssim(img1, img2, data_range=max_val, multichannel=True, channel_axis=2)
        else:  # Grayscale
            return ssim(img1[0], img2[0], data_range=max_val)
    else:  # 2D image
        return ssim(img1, img2, data_range=max_val)

def evaluate_model_metrics(generator, dataloader, device, max_val=1.0):
    """
    Evaluate model using SSIM and PSNR metrics
    
    Args:
        generator: trained generator model
        dataloader: validation dataloader
        device: torch device
        max_val: maximum pixel value (1.0 for normalized images)
    
    Returns:
        dict: {'psnr': float, 'ssim': float}
    """
    generator.eval()
    
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for sketches, photos in tqdm(dataloader, desc="Evaluating metrics"):
            sketches = sketches.to(device)
            photos = photos.to(device)
            
            # Generate fake photos
            fake_photos = generator(sketches)
            
            # Convert to numpy and calculate metrics
            fake_np = fake_photos.cpu().numpy()
            real_np = photos.cpu().numpy()
            
            # Calculate PSNR for each image in batch
            for i in range(fake_np.shape[0]):
                psnr_val = calculate_psnr(fake_np[i], real_np[i], max_val)
                psnr_values.append(psnr_val)
                
                ssim_val = calculate_ssim(fake_np[i], real_np[i], max_val)
                ssim_values.append(ssim_val)
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'psnr_std': np.std(psnr_values),
        'ssim_std': np.std(ssim_values)
    }

def plot_evaluation_results(metrics, output_dir):
    """
    Create plots showing SSIM and PSNR results
    
    Args:
        metrics: dict with 'psnr', 'ssim', 'psnr_std', 'ssim_std'
        output_dir: path to save plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR plot
    ax1.bar(['PSNR'], [metrics['psnr']], yerr=[metrics['psnr_std']], 
            capsize=10, color='skyblue', edgecolor='navy', linewidth=2)
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title(f'Peak Signal-to-Noise Ratio\n{metrics["psnr"]:.2f} ± {metrics["psnr_std"]:.2f} dB')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(30, metrics['psnr'] + metrics['psnr_std'] + 5))
    
    # Add value text on bar
    ax1.text(0, metrics['psnr'] + metrics['psnr_std'] + 1, 
             f'{metrics["psnr"]:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # SSIM plot
    ax2.bar(['SSIM'], [metrics['ssim']], yerr=[metrics['ssim_std']], 
            capsize=10, color='lightcoral', edgecolor='darkred', linewidth=2)
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'Structural Similarity Index\n{metrics["ssim"]:.4f} ± {metrics["ssim_std"]:.4f}')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add value text on bar
    ax2.text(0, metrics['ssim'] + metrics['ssim_std'] + 0.05, 
             f'{metrics["ssim"]:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"PSNR: {metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    print(f"{'='*50}")

def compare_sample_images_with_metrics(generator, dataloader, device, output_dir, num_samples=5):
    """
    Generate sample images and calculate metrics for each
    
    Args:
        generator: trained generator model
        dataloader: validation dataloader  
        device: torch device
        output_dir: path to save results
        num_samples: number of sample images to show
    """
    generator.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for batch_idx, (sketches, photos) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            sketches = sketches.to(device)
            photos = photos.to(device)
            
            fake_photos = generator(sketches)
            
            # Take first image from batch
            sketch = sketches[0].cpu()
            real = photos[0].cpu() 
            fake = fake_photos[0].cpu()
            
            # Calculate metrics for this sample
            psnr_val = calculate_psnr(fake.numpy(), real.numpy())
            ssim_val = calculate_ssim(fake.numpy(), real.numpy())
            
            # Convert tensors to displayable format
            sketch_np = (sketch.permute(1, 2, 0) + 1) / 2  # Denormalize from [-1,1] to [0,1]
            real_np = (real.permute(1, 2, 0) + 1) / 2
            fake_np = (fake.permute(1, 2, 0) + 1) / 2
            
            # Plot images
            axes[batch_idx, 0].imshow(sketch_np)
            axes[batch_idx, 0].set_title(f'Input Sketch')
            axes[batch_idx, 0].axis('off')
            
            axes[batch_idx, 1].imshow(real_np)
            axes[batch_idx, 1].set_title(f'Real Photo')
            axes[batch_idx, 1].axis('off')
            
            axes[batch_idx, 2].imshow(fake_np)
            axes[batch_idx, 2].set_title(f'Generated Photo\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}')
            axes[batch_idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_comparison_with_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()