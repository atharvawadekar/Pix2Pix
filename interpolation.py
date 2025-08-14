import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from models.generator import Generator
from datasets.cuhk_dataset import load_cuhk_dataset, CUHKFaceSketchDataset

class LatentSpaceInterpolator:
    """
    Helper class for performing latent space interpolation with your specific generator
    """
    def __init__(self, generator):
        self.generator = generator
        
    def interpolate_in_bottleneck(self, sketch1, sketch2, num_steps=8):
        """
        Interpolate in the bottleneck layer (d8 - deepest features)
        """
        self.generator.eval()
        device = sketch1.device
        
        with torch.no_grad():
            # Get features for both sketches
            _, features1 = self.generator(sketch1, return_features=True)
            _, features2 = self.generator(sketch2, return_features=True)
            
            # Interpolate in bottleneck (d8)
            bottleneck1 = features1['d8']
            bottleneck2 = features2['d8']
            
            alphas = torch.linspace(0, 1, num_steps).to(device)
            interpolated_results = []
            
            for alpha in alphas:
                # Interpolate bottleneck features
                interpolated_bottleneck = (1 - alpha) * bottleneck1 + alpha * bottleneck2
                
                # Enable feature injection
                self.generator.enable_feature_injection('d8', interpolated_bottleneck)
                
                # Forward pass with injected features (use sketch1 as base input)
                result = self.generator(sketch1)
                interpolated_results.append(result)
                
                # Disable injection for next iteration
                self.generator.disable_feature_injection()
            
            return torch.cat(interpolated_results, dim=0)
    
    def interpolate_at_layer(self, sketch1, sketch2, layer_name='d4', num_steps=8):
        """
        Interpolate at any specified encoder layer
        Available layers: d1, d2, d3, d4, d5, d6, d7, d8
        """
        self.generator.eval()
        device = sketch1.device
        
        with torch.no_grad():
            # Get features for both sketches
            _, features1 = self.generator(sketch1, return_features=True)
            _, features2 = self.generator(sketch2, return_features=True)
            
            # Interpolate at specified layer
            feature1 = features1[layer_name]
            feature2 = features2[layer_name]
            
            alphas = torch.linspace(0, 1, num_steps).to(device)
            interpolated_results = []
            
            for alpha in alphas:
                # Interpolate features
                interpolated_feature = (1 - alpha) * feature1 + alpha * feature2
                
                # Enable feature injection
                self.generator.enable_feature_injection(layer_name, interpolated_feature)
                
                # Forward pass with injected features
                result = self.generator(sketch1)
                interpolated_results.append(result)
                
                # Disable injection
                self.generator.disable_feature_injection()
            
            return torch.cat(interpolated_results, dim=0)
    
    def multi_layer_interpolation(self, sketch1, sketch2, layers=['d8', 'd6', 'd4', 'd2'], num_steps=6):
        """
        Perform interpolation at multiple layers to show different levels of feature mixing
        """
        results = {}
        
        for layer in layers:
            results[layer] = self.interpolate_at_layer(sketch1, sketch2, layer, num_steps)
        
        return results

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for plotting"""
    tensor = tensor.cpu().squeeze(0)
    # Denormalize from [-1,1] to [0,1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    if tensor.shape[0] == 1:  # Grayscale
        return tensor.squeeze(0).numpy()
    elif tensor.shape[0] == 3:  # RGB
        return tensor.permute(1, 2, 0).numpy()
    else:
        raise ValueError(f"Unsupported number of channels: {tensor.shape[0]}")

def create_latent_interpolation_plot(interpolator, sketch1, sketch2, layer_name='d8', num_steps=8, save_path=None):
    """
    Create a single interpolation plot between two sketches
    """
    # Perform interpolation
    interpolated_results = interpolator.interpolate_at_layer(sketch1, sketch2, layer_name, num_steps)
    
    # Create plot
    fig, axes = plt.subplots(1, num_steps + 2, figsize=(20, 4))
    
    # Plot input sketch 1
    sketch1_np = tensor_to_numpy(sketch1)
    axes[0].imshow(sketch1_np, cmap='gray')
    axes[0].set_title('Input Sketch 1')
    axes[0].axis('off')
    
    # Plot interpolation steps
    for i in range(num_steps):
        result_np = tensor_to_numpy(interpolated_results[i:i+1])
        axes[i+1].imshow(result_np, cmap='gray')
        alpha = i / (num_steps - 1)
        axes[i+1].set_title(f'α={alpha:.2f}')
        axes[i+1].axis('off')
    
    # Plot input sketch 2
    sketch2_np = tensor_to_numpy(sketch2)
    axes[-1].imshow(sketch2_np, cmap='gray')
    axes[-1].set_title('Input Sketch 2')
    axes[-1].axis('off')
    
    plt.suptitle(f'Latent Space Interpolation at Layer: {layer_name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Interpolation plot saved to: {save_path}")
    
    plt.show()

def create_multi_layer_comparison(interpolator, sketch1, sketch2, layers=['d8', 'd6', 'd4', 'd2'], 
                                num_steps=6, save_path=None):
    """
    Create comparison plot showing interpolation at different layers
    """
    # Get interpolation results for all layers
    layer_results = interpolator.multi_layer_interpolation(sketch1, sketch2, layers, num_steps)
    
    # Create plot
    fig, axes = plt.subplots(len(layers), num_steps + 2, figsize=(20, 5 * len(layers)))
    
    for i, layer in enumerate(layers):
        results = layer_results[layer]
        
        # Plot input sketch 1
        sketch1_np = tensor_to_numpy(sketch1)
        axes[i, 0].imshow(sketch1_np, cmap='gray')
        axes[i, 0].set_title(f'{layer}\nInput 1')
        axes[i, 0].axis('off')
        
        # Plot interpolation results
        for j in range(num_steps):
            result_np = tensor_to_numpy(results[j:j+1])
            axes[i, j+1].imshow(result_np, cmap='gray')
            alpha = j / (num_steps - 1)
            axes[i, j+1].set_title(f'α={alpha:.2f}')
            axes[i, j+1].axis('off')
        
        # Plot input sketch 2
        sketch2_np = tensor_to_numpy(sketch2)
        axes[i, -1].imshow(sketch2_np, cmap='gray')
        axes[i, -1].set_title('Input 2')
        axes[i, -1].axis('off')
    
    plt.suptitle('Multi-Layer Latent Space Interpolation Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-layer comparison saved to: {save_path}")
    
    plt.show()

def create_bottleneck_interpolation_detailed(interpolator, sketch1, sketch2, num_steps=12, save_path=None):
    """
    Create detailed bottleneck interpolation with more steps
    """
    # Perform bottleneck interpolation
    bottleneck_results = interpolator.interpolate_in_bottleneck(sketch1, sketch2, num_steps)
    
    # Create plot
    fig, axes = plt.subplots(2, num_steps // 2 + 1, figsize=(20, 8))
    axes = axes.flatten()
    
    # Plot input sketch 1
    sketch1_np = tensor_to_numpy(sketch1)
    axes[0].imshow(sketch1_np, cmap='gray')
    axes[0].set_title('Input Sketch 1')
    axes[0].axis('off')
    
    # Plot interpolation steps
    for i in range(num_steps):
        result_np = tensor_to_numpy(bottleneck_results[i:i+1])
        axes[i+1].imshow(result_np, cmap='gray')
        alpha = i / (num_steps - 1)
        axes[i+1].set_title(f'Step {i+1}\nα={alpha:.2f}')
        axes[i+1].axis('off')
    
    # Plot input sketch 2
    sketch2_np = tensor_to_numpy(sketch2)
    axes[-1].imshow(sketch2_np, cmap='gray')
    axes[-1].set_title('Input Sketch 2')
    axes[-1].axis('off')
    
    plt.suptitle('Detailed Bottleneck Layer Interpolation (Semantic Features)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed bottleneck interpolation saved to: {save_path}")
    
    plt.show()

def demonstrate_all_interpolations(generator_path, data_root, output_dir, img_size=256, num_samples=5):
    """
    Main function to demonstrate all types of latent space interpolations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading generator...")
    generator = Generator(in_channels=1, out_channels=1).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    print("Generator loaded successfully!")
    
    # Load dataset
    print("Loading dataset...")
    sketches, photos = load_cuhk_dataset(data_root, size=img_size)
    val_dataset = CUHKFaceSketchDataset(sketches, photos, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=num_samples, shuffle=True)
    
    # Get sample sketches
    sketches_batch, _ = next(iter(val_loader))
    sketches_batch = sketches_batch.to(device)
    
    # Create interpolator
    interpolator = LatentSpaceInterpolator(generator)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Demonstrate different interpolations
    for i in range(min(2, len(sketches_batch) - 1)):  # Use pairs of sketches
        sketch1 = sketches_batch[i:i+1]
        sketch2 = sketches_batch[i+1:i+2]
        
        print(f"\nCreating interpolations for sketch pair {i+1}...")
        
        # 1. Single layer interpolation (bottleneck)
        create_latent_interpolation_plot(
            interpolator, sketch1, sketch2, 
            layer_name='d8', num_steps=8,
            save_path=output_dir / f'bottleneck_interpolation_pair_{i+1}.png'
        )
        
        # 2. Multi-layer comparison
        create_multi_layer_comparison(
            interpolator, sketch1, sketch2,
            layers=['d8', 'd6', 'd4', 'd2'], num_steps=6,
            save_path=output_dir / f'multi_layer_comparison_pair_{i+1}.png'
        )
        
        # 3. Detailed bottleneck interpolation
        create_bottleneck_interpolation_detailed(
            interpolator, sketch1, sketch2, num_steps=10,
            save_path=output_dir / f'detailed_bottleneck_pair_{i+1}.png'
        )
    
    print(f"\n✅ All interpolation visualizations completed!")
    print(f"📁 Check the results in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate Latent Space Interpolations for Pix2Pix')
    parser.add_argument('--generator_path', type=str, required=True, 
                       help='Path to trained generator weights (e.g., ./outputs/generator_final.pth)')
    parser.add_argument('--data_root', type=str, required=True, 
                       help='Path to CUHK dataset')
    parser.add_argument('--output_dir', type=str, default='./latent_interpolation_results', 
                       help='Output directory for interpolation results')
    parser.add_argument('--img_size', type=int, default=256, 
                       help='Image size')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='Number of samples to load')
    
    args = parser.parse_args()
    
    # Run interpolation demonstration
    demonstrate_all_interpolations(
        generator_path=args.generator_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        img_size=args.img_size,
        num_samples=args.num_samples
    )

if __name__ == '__main__':
    main()

# Example usage:
# python latent_interpolation.py --generator_path ./outputs/generator_final.pth --data_root /path/to/CUHK/dataset