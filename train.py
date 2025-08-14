import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from models.generator import Generator
from models.discriminator import Discriminator
from models.losses import GANLoss
from datasets.cuhk_dataset import load_cuhk_dataset, CUHKFaceSketchDataset
from visualize import save_sample_images
from metrics import evaluate_model_metrics 

def train_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, gan_loss, l1_loss, device, lambda_l1=100):
    
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    
    for i, (sketches, photos) in enumerate(tqdm(dataloader, desc="Training")):
        sketches = sketches.to(device)
        photos = photos.to(device)
        
        
        d_optimizer.zero_grad()
        
       
        real_pred = discriminator(sketches, photos)
        d_real_loss = gan_loss(real_pred, True)
        
   
        fake_photos = generator(sketches)
        fake_pred = discriminator(sketches, fake_photos.detach())
        d_fake_loss = gan_loss(fake_pred, False)
        
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()
        d_optimizer.step()
        
       
        g_optimizer.zero_grad()
        
        fake_pred = discriminator(sketches, fake_photos)
        g_gan_loss = gan_loss(fake_pred, True)
        g_l1_loss = l1_loss(fake_photos, photos)
        
        g_loss = g_gan_loss + lambda_l1 * g_l1_loss
        g_loss.backward()
        g_optimizer.step()
        
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
    
    return total_g_loss / len(dataloader), total_d_loss / len(dataloader)

def validate(generator, dataloader, l1_loss, device):
    
    generator.eval()
    total_l1_loss = 0
    
    with torch.no_grad():
        for sketches, photos in tqdm(dataloader, desc="Validating"):
            sketches = sketches.to(device)
            photos = photos.to(device)
            
            fake_photos = generator(sketches)
            l1_loss_val = l1_loss(fake_photos, photos)
            total_l1_loss += l1_loss_val.item()
    
    return total_l1_loss / len(dataloader)

def evaluate_metrics_per_epoch(generator, dataloader, device, max_samples=500):
    """
    ssim qnd psnr for a subset of images
    """
    generator.eval()
    
 
    if max_samples < len(dataloader.dataset):
        batches_needed = min(max_samples // dataloader.batch_size + 1, len(dataloader))
        
        subset_data = []
        for i, batch in enumerate(dataloader):
            if i >= batches_needed:
                break
            subset_data.append(batch)
        
        from torch.utils.data import TensorDataset
        
        all_sketches = torch.cat([batch[0] for batch in subset_data], dim=0)[:max_samples]
        all_photos = torch.cat([batch[1] for batch in subset_data], dim=0)[:max_samples]
        
        subset_dataset = TensorDataset(all_sketches, all_photos)
        subset_loader = DataLoader(subset_dataset, batch_size=dataloader.batch_size, shuffle=False)
    else:
        subset_loader = dataloader
    
    metrics = evaluate_model_metrics(generator, subset_loader, device, max_val=1.0)
    
    print(f"Raw metrics from evaluate_model_metrics: {metrics}")
    
    return {
        'ssim_mean': metrics['ssim'],
        'ssim_std': metrics['ssim_std'],
        'psnr_mean': metrics['psnr'],
        'psnr_std': metrics['psnr_std']
    }

def plot_training_curves_with_metrics(history, output_dir):
    """
    plot training curves including metrics
    """
    epochs = range(1, len(history['g_loss']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].plot(epochs, history['g_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Generator Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='y', labelcolor='blue')
    
    axes[0, 1].plot(epochs, history['d_loss'], 'r-', linewidth=2)
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Discriminator Loss')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='y', labelcolor='red')
    
    axes[0, 2].plot(epochs, history['val_l1_loss'], 'purple', linewidth=2, marker='^', markersize=4)
    axes[0, 2].set_title('Validation L1 Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('L1 Loss')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].tick_params(axis='y', labelcolor='purple')
    
 
    if 'ssim_mean' in history and len(history['ssim_mean']) > 0:
        metric_epochs = range(1, len(history['ssim_mean']) + 1)
        axes[1, 0].plot(metric_epochs, history['ssim_mean'], 'g-', linewidth=2, marker='o', markersize=4)
        axes[1, 0].fill_between(metric_epochs, 
                               [mean - std for mean, std in zip(history['ssim_mean'], history['ssim_std'])],
                               [mean + std for mean, std in zip(history['ssim_mean'], history['ssim_std'])],
                               alpha=0.2, color='green')
        axes[1, 0].set_title('SSIM Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='y', labelcolor='green')
    else:
        axes[1, 0].text(0.5, 0.5, 'No SSIM data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('SSIM Score (No Data)')
    
    
    if 'psnr_mean' in history and len(history['psnr_mean']) > 0:
        metric_epochs = range(1, len(history['psnr_mean']) + 1)
        axes[1, 1].plot(metric_epochs, history['psnr_mean'], 'orange', linewidth=2, marker='s', markersize=4)
        axes[1, 1].fill_between(metric_epochs,
                               [mean - std for mean, std in zip(history['psnr_mean'], history['psnr_std'])],
                               [mean + std for mean, std in zip(history['psnr_mean'], history['psnr_std'])],
                               alpha=0.2, color='orange')
        axes[1, 1].set_title('PSNR Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('PSNR (dB)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='y', labelcolor='orange')
    else:
        axes[1, 1].text(0.5, 0.5, 'No PSNR data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('PSNR Score (No Data)')
    
    axes[1, 2].plot(epochs, history['g_loss'], 'b-', linewidth=2, alpha=0.7, label='Generator')
    axes[1, 2].plot(epochs, history['d_loss'], 'r-', linewidth=2, alpha=0.7, label='Discriminator')
    axes[1, 2].set_title('Loss Comparison')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.suptitle('Training Progress with Metrics', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_with_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    create_individual_loss_plots(history, output_dir)

def create_individual_loss_plots(history, output_dir):
    """
    individual loss plots for generator, discriminator, validation L1 loss, SSIM and PSNR
    """
    epochs = range(1, len(history['g_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['g_loss'], 'b-', linewidth=2, marker='o', markersize=4)
    plt.title('Generator Loss Over Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Generator Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().tick_params(axis='y', labelcolor='blue')
    plt.gca().spines['left'].set_color('blue')
    plt.tight_layout()
    plt.savefig(output_dir / 'generator_loss_individual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['d_loss'], 'r-', linewidth=2, marker='s', markersize=4)
    plt.title('Discriminator Loss Over Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Discriminator Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().tick_params(axis='y', labelcolor='red')
    plt.gca().spines['left'].set_color('red')
    plt.tight_layout()
    plt.savefig(output_dir / 'discriminator_loss_individual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_l1_loss'], 'purple', linewidth=2, marker='^', markersize=4)
    plt.title('Validation L1 Loss Over Training', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L1 Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().tick_params(axis='y', labelcolor='purple')
    plt.gca().spines['left'].set_color('purple')
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_l1_loss_individual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'ssim_mean' in history and len(history['ssim_mean']) > 0:
        metric_epochs = range(1, len(history['ssim_mean']) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(metric_epochs, history['ssim_mean'], 'g-', linewidth=2, marker='o', markersize=4)
        plt.fill_between(metric_epochs, 
                        [mean - std for mean, std in zip(history['ssim_mean'], history['ssim_std'])],
                        [mean + std for mean, std in zip(history['ssim_mean'], history['ssim_std'])],
                        alpha=0.2, color='green')
        plt.title('SSIM Score Over Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('SSIM', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.gca().tick_params(axis='y', labelcolor='green')
        plt.gca().spines['left'].set_color('green')
        plt.tight_layout()
        plt.savefig(output_dir / 'ssim_individual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'psnr_mean' in history and len(history['psnr_mean']) > 0:
        metric_epochs = range(1, len(history['psnr_mean']) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(metric_epochs, history['psnr_mean'], 'orange', linewidth=2, marker='s', markersize=4)
        plt.fill_between(metric_epochs,
                        [mean - std for mean, std in zip(history['psnr_mean'], history['psnr_std'])],
                        [mean + std for mean, std in zip(history['psnr_mean'], history['psnr_std'])],
                        alpha=0.2, color='orange')
        plt.title('PSNR Score Over Training', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().tick_params(axis='y', labelcolor='orange')
        plt.gca().spines['left'].set_color('orange')
        plt.tight_layout()
        plt.savefig(output_dir / 'psnr_individual.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f" Individual loss plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix for Face Sketch to Photo Translation')
    parser.add_argument('--data_root', type=str, required=True, help='Path to CUHK dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100, help='L1 loss weight')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--metric_freq', type=int, default=5, help='Frequency to evaluate metrics (every N epochs)')
    parser.add_argument('--metric_samples', type=int, default=300, help='Number of samples for metric evaluation')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'samples').mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading CUHK dataset with augmentation...")
    sketches, photos = load_cuhk_dataset(args.data_root, size=args.img_size)
    
    train_dataset = CUHKFaceSketchDataset(sketches, photos, mode='train')
    val_dataset = CUHKFaceSketchDataset(sketches, photos, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    gan_loss = GANLoss().to(device)
    l1_loss = nn.L1Loss()
    
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100)
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100)
    
    history = {
        'g_loss': [],
        'd_loss': [],
        'val_l1_loss': [],
        'ssim_mean': [],
        'ssim_std': [],
        'psnr_mean': [],
        'psnr_std': []
    }
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Metrics will be evaluated every {args.metric_freq} epochs using {args.metric_samples} samples")
    
    for epoch in range(args.epochs):
        g_loss, d_loss = train_epoch(generator, discriminator, train_loader,g_optimizer, d_optimizer, gan_loss, l1_loss,device, args.lambda_l1)
        
        val_l1_loss = validate(generator, val_loader, l1_loss, device)
        
        g_scheduler.step()
        d_scheduler.step()
        
        history['g_loss'].append(g_loss)
        history['d_loss'].append(d_loss)
        history['val_l1_loss'].append(val_l1_loss)
        
        if (epoch + 1) % args.metric_freq == 0 or epoch == args.epochs - 1:
            print(f"\nEvaluating metrics for epoch {epoch + 1}...")
            try:
                metrics = evaluate_metrics_per_epoch(generator, val_loader, device, args.metric_samples)
                print(f"DEBUG: Metrics returned: {metrics}")
                
                if metrics and 'ssim_mean' in metrics:
                    history['ssim_mean'].append(metrics['ssim_mean'])
                    history['ssim_std'].append(metrics['ssim_std'])
                    history['psnr_mean'].append(metrics['psnr_mean'])
                    history['psnr_std'].append(metrics['psnr_std'])
                    
                    print(f"Epoch [{epoch+1}/{args.epochs}] - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, Val L1: {val_l1_loss:.4f}")
                    print(f"SSIM: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}, PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
                else:
                    print(f"ERROR: Metrics evaluation returned invalid data: {metrics}")
                    print(f"Epoch [{epoch+1}/{args.epochs}] - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, Val L1: {val_l1_loss:.4f}")
                    
            except Exception as e:
                print(f"ERROR evaluating metrics: {e}")
                import traceback
                traceback.print_exc()
                print(f"Epoch [{epoch+1}/{args.epochs}] - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, Val L1: {val_l1_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}] - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, Val L1: {val_l1_loss:.4f}")
        
        if (epoch + 1) % args.save_freq == 0:
            save_sample_images(generator, val_loader, device, epoch + 1, output_dir / 'samples')
        
        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_loss': g_loss,
                'd_loss': d_loss,
                'val_l1_loss': val_l1_loss,
                'history': history  # Save full history in checkpoints
            }, output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth')
    
    torch.save(generator.state_dict(), output_dir / 'generator_final.pth')
    torch.save(discriminator.state_dict(), output_dir / 'discriminator_final.pth')
    
    with open(output_dir / 'training_history_with_metrics.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_training_curves_with_metrics(history, output_dir)
    
    print("Training completed!")
    print(f"Models saved to: {output_dir}")
    print(f"Sample images saved to: {output_dir / 'samples'}")
    print(f"Training curves with metrics saved to: {output_dir / 'training_curves_with_metrics.png'}")
    print(f"Individual loss plots saved to: {output_dir}/")
    print(f"  - generator_loss_individual.png")
    print(f"  - discriminator_loss_individual.png") 
    print(f"  - validation_l1_loss_individual.png")
    if len(history['ssim_mean']) > 0:
        print(f"  - ssim_individual.png")
    if len(history['psnr_mean']) > 0:
        print(f"  - psnr_individual.png")
    print(f"Complete history saved to: {output_dir / 'training_history_with_metrics.json'}")

if __name__ == '__main__':
    main()