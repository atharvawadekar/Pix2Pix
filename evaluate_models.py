import os
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from models.generator import Generator
from datasets.cuhk_dataset import load_cuhk_dataset, CUHKFaceSketchDataset
from metrics import evaluate_model_metrics, plot_evaluation_results, compare_sample_images_with_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained Pix2Pix model with SSIM and PSNR')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved generator model (.pth file)')
    parser.add_argument('--data_root', type=str, required=True, help='Path to CUHK dataset')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of sample images to show')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading CUHK dataset...")
    sketches, photos = load_cuhk_dataset(args.data_root, size=args.img_size)
    val_dataset = CUHKFaceSketchDataset(sketches, photos, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    generator = Generator().to(device)
    
    # Load state dict
    if args.model_path.endswith('.pth'):
        # If it's a direct state dict
        generator.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        # If it's a checkpoint with multiple components
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
    
    generator.eval()
    print("Model loaded successfully!")
    
    # Evaluate metrics
    print("\nEvaluating model with SSIM and PSNR metrics...")
    metrics = evaluate_model_metrics(generator, val_loader, device, max_val=1.0)
    
    # Create plots
    plot_evaluation_results(metrics, output_dir)
    
    # Create sample comparisons
    compare_sample_images_with_metrics(generator, val_loader, device, output_dir, num_samples=args.num_samples)
    
    # Save detailed results
    import json
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump({
            'model_path': str(args.model_path),
            'dataset_path': str(args.data_root),
            'metrics': metrics,
            'parameters': {
                'batch_size': args.batch_size,
                'img_size': args.img_size,
                'num_samples': args.num_samples
            }
        }, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Metrics file: {output_dir / 'evaluation_results.json'}")

if __name__ == '__main__':
    main()