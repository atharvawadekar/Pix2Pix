import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

def denormalize_tensor(tensor):
    """Convert tensor from [-1, 1] to [0, 1] range for visualization."""
    return (tensor + 1.0) / 2.0

def save_sample_images(generator, dataloader, device, epoch, output_dir, num_samples=8):
    """Save sample images showing sketch → generated → real progression.
    
    Args:
        generator: Trained generator model
        dataloader: Validation dataloader
        device: Device (CPU/GPU)
        epoch: Current epoch number
        output_dir: Directory to save images
        num_samples: Number of samples to visualize
    """
    generator.eval()
    
    with torch.no_grad():
        # Get a batch of validation data
        sketches, photos = next(iter(dataloader))
        sketches = sketches.to(device)
        photos = photos.to(device)
        
        # Generate fake photos
        fake_photos = generator(sketches)
        
        # Take only the specified number of samples
        sketches = sketches[:num_samples]
        fake_photos = fake_photos[:num_samples]
        photos = photos[:num_samples]
        
        # Denormalize images for visualization
        sketches_vis = denormalize_tensor(sketches)
        fake_photos_vis = denormalize_tensor(fake_photos)
        photos_vis = denormalize_tensor(photos)
        
        # Create comparison grid: [sketches, generated, real]
        comparison = torch.cat([sketches_vis, fake_photos_vis, photos_vis], dim=0)
        
        # Save grid image
        output_path = Path(output_dir) / f"epoch_{epoch:03d}.png"
        vutils.save_image(
            comparison, 
            output_path, 
            nrow=num_samples, 
            normalize=False,
            padding=2,
            pad_value=1.0  # White padding
        )
        
        print(f"Sample images saved to: {output_path}")

def create_comparison_grid(sketches, generated, real, num_samples=8, figsize=(15, 6)):
    """Create a matplotlib comparison grid.
    
    Args:
        sketches: Sketch tensors [B, 1, H, W]
        generated: Generated photo tensors [B, 1, H, W]
        real: Real photo tensors [B, 1, H, W]
        num_samples: Number of samples to show
        figsize: Figure size
        
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(3, num_samples, figsize=figsize)
    
    # Convert tensors to numpy and denormalize
    sketches_np = denormalize_tensor(sketches[:num_samples]).cpu().numpy()
    generated_np = denormalize_tensor(generated[:num_samples]).cpu().numpy()
    real_np = denormalize_tensor(real[:num_samples]).cpu().numpy()
    
    # Plot each sample
    for i in range(num_samples):
        # Sketch (top row)
        axes[0, i].imshow(sketches_np[i, 0], cmap='gray')
        axes[0, i].set_title('Sketch' if i == 0 else '')
        axes[0, i].axis('off')
        
        # Generated (middle row)
        axes[1, i].imshow(generated_np[i, 0], cmap='gray')
        axes[1, i].set_title('Generated' if i == 0 else '')
        axes[1, i].axis('off')
        
        # Real (bottom row)
        axes[2, i].imshow(real_np[i, 0], cmap='gray')
        axes[2, i].set_title('Real' if i == 0 else '')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    return fig

def save_training_progress(generator, val_loader, device, output_dir, epoch):
    """Save detailed training progress with multiple visualizations.
    
    Args:
        generator: Generator model
        val_loader: Validation dataloader
        device: Device
        output_dir: Output directory
        epoch: Current epoch
    """
    generator.eval()
    output_dir = Path(output_dir)
    
    with torch.no_grad():
        sketches, photos = next(iter(val_loader))
        sketches = sketches.to(device)
        photos = photos.to(device)
        
        generated = generator(sketches)
        
        # Create comparison grid
        fig = create_comparison_grid(sketches, generated, photos)
        
        # Save matplotlib figure
        plt_path = output_dir / f"comparison_epoch_{epoch:03d}.png"
        fig.savefig(plt_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save individual images for detailed inspection
        individual_dir = output_dir / f"epoch_{epoch:03d}_individual"
        individual_dir.mkdir(exist_ok=True)
        
        for i in range(min(4, len(sketches))):  # Save first 4 samples
            sketch_img = denormalize_tensor(sketches[i:i+1])
            gen_img = denormalize_tensor(generated[i:i+1])
            real_img = denormalize_tensor(photos[i:i+1])
            
            vutils.save_image(sketch_img, individual_dir / f"sample_{i:02d}_sketch.png")
            vutils.save_image(gen_img, individual_dir / f"sample_{i:02d}_generated.png")
            vutils.save_image(real_img, individual_dir / f"sample_{i:02d}_real.png")

def plot_training_curves(history, output_path):
    """Plot and save training curves.
    
    Args:
        history: Dictionary containing training history
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator and Discriminator losses
    axes[0, 0].plot(history['g_loss'], label='Generator Loss', color='blue', alpha=0.8)
    axes[0, 0].plot(history['d_loss'], label='Discriminator Loss', color='red', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('GAN Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation L1 Loss
    axes[0, 1].plot(history['val_l1_loss'], label='Validation L1 Loss', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('L1 Loss')
    axes[0, 1].set_title('Validation L1 Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss ratio (G/D balance)
    g_loss = np.array(history['g_loss'])
    d_loss = np.array(history['d_loss'])
    loss_ratio = g_loss / (d_loss + 1e-8)  # Avoid division by zero
    
    axes[1, 0].plot(loss_ratio, label='G/D Loss Ratio', color='purple')
    axes[1, 0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal Balance')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].set_title('Generator/Discriminator Loss Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Smoothed losses (moving average)
    window = 10
    if len(history['g_loss']) > window:
        g_smooth = np.convolve(g_loss, np.ones(window)/window, mode='valid')
        d_smooth = np.convolve(d_loss, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(history['val_l1_loss'], np.ones(window)/window, mode='valid')
        
        axes[1, 1].plot(g_smooth, label='Generator (smoothed)', alpha=0.8)
        axes[1, 1].plot(d_smooth, label='Discriminator (smoothed)', alpha=0.8)
        axes[1, 1].plot(val_smooth, label='Validation L1 (smoothed)', alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title(f'Smoothed Losses (window={window})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_feature_maps(generator, input_tensor, layer_name='down4', output_dir='./feature_maps'):
    """Visualize intermediate feature maps from the generator.
    
    Args:
        generator: Generator model
        input_tensor: Input sketch tensor
        layer_name: Name of layer to visualize
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Hook to capture feature maps
    features = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    for name, module in generator.named_modules():
        if layer_name in name:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    generator.eval()
    with torch.no_grad():
        _ = generator(input_tensor)
    
    # Visualize feature maps
    for name, feature_map in features.items():
        # Take first sample and first few channels
        fm = feature_map[0, :16]  # First 16 channels
        
        # Create grid
        grid = vutils.make_grid(fm.unsqueeze(1), nrow=4, normalize=True, padding=2)
        
        # Save
        output_path = output_dir / f"{name}_features.png"
        vutils.save_image(grid, output_path)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

def create_interpolation_video(generator, sketch1, sketch2, device, output_path, num_frames=30):
    """Create interpolation video between two sketches.
    
    Args:
        generator: Generator model
        sketch1: First sketch tensor
        sketch2: Second sketch tensor
        device: Device
        output_path: Path to save video
        num_frames: Number of interpolation frames
    """
    generator.eval()
    
    # Create interpolation weights
    alphas = np.linspace(0, 1, num_frames)
    
    frames = []
    with torch.no_grad():
        for alpha in alphas:
            # Linear interpolation in input space
            interpolated_sketch = (1 - alpha) * sketch1 + alpha * sketch2
            
            # Generate photo
            generated_photo = generator(interpolated_sketch.to(device))
            
            # Convert to numpy
            frame = denormalize_tensor(generated_photo).cpu().squeeze().numpy()
            frames.append((frame * 255).astype(np.uint8))
    
    # Save as video (requires OpenCV)
    height, width = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, 10.0, (width, height), isColor=False)
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Interpolation video saved to: {output_path}")

def plot_images(real_img, sketch_img, generated_img=None, figsize=(12, 4)):
    """Plot comparison of sketch, generated, and real images.
    
    Args:
        real_img: Real photo tensor or numpy array
        sketch_img: Sketch tensor or numpy array
        generated_img: Generated photo tensor or numpy array (optional)
        figsize: Figure size
    """
    num_plots = 3 if generated_img is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(sketch_img):
        sketch_img = denormalize_tensor(sketch_img).squeeze().cpu().numpy()
    if torch.is_tensor(real_img):
        real_img = denormalize_tensor(real_img).squeeze().cpu().numpy()
    if generated_img is not None and torch.is_tensor(generated_img):
        generated_img = denormalize_tensor(generated_img).squeeze().cpu().numpy()
    
    # Plot sketch
    axes[0].imshow(sketch_img, cmap='gray')
    axes[0].set_title('Sketch')
    axes[0].axis('off')
    
    # Plot generated (if provided)
    if generated_img is not None:
        axes[1].imshow(generated_img, cmap='gray')
        axes[1].set_title('Generated')
        axes[1].axis('off')
        
        # Plot real
        axes[2].imshow(real_img, cmap='gray')
        axes[2].set_title('Real')
        axes[2].axis('off')
    else:
        # Plot real
        axes[1].imshow(real_img, cmap='gray')
        axes[1].set_title('Real')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()