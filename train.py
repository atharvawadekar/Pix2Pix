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

def train_epoch(generator, discriminator, dataloader, g_optimizer, d_optimizer, gan_loss, l1_loss, device, lambda_l1=100):
    #single epoch training
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    
    for i, (sketches, photos) in enumerate(tqdm(dataloader, desc="Training")):#tqdm for loading
        sketches = sketches.to(device)
        photos = photos.to(device)
        
        #train discriminator
        d_optimizer.zero_grad()
        
        #real pairs
        real_pred = discriminator(sketches, photos)
        d_real_loss = gan_loss(real_pred, True)
        
        #fake pairs
        fake_photos = generator(sketches)
        fake_pred = discriminator(sketches, fake_photos.detach())
        d_fake_loss = gan_loss(fake_pred, False)
        
        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()
        d_optimizer.step()
        
        #train Generator
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
    """Validate the model."""
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

def main():
    #get all parameters as via input
    parser = argparse.ArgumentParser(description='Train Pix2Pix for Face Sketch to Photo Translation')
    parser.add_argument('--data_root', type=str, required=True, help='Path to CUHK dataset')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--lambda_l1', type=float, default=100, help='L1 loss weight')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    
    args = parser.parse_args()
    
    #Output locations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'samples').mkdir(exist_ok=True)
    
    #cuda for speed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #preprocess data
    print("Loading CUHK dataset with augmentation...")
    sketches, photos = load_cuhk_dataset(args.data_root, size=args.img_size)
    
    #dataloaders
    train_dataset = CUHKFaceSketchDataset(sketches, photos, mode='train')
    val_dataset = CUHKFaceSketchDataset(sketches, photos, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Loss functions
    gan_loss = GANLoss().to(device)
    l1_loss = nn.L1Loss()
    
    #defining optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    #LR scheduler, for 100 epochs LR is 1 and then it starts reducing
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100)
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 100) / 100)
    
    #storing losses
    history = {
        'g_loss': [],
        'd_loss': [],
        'val_l1_loss': []
    }
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        #training
        g_loss, d_loss = train_epoch(generator, discriminator, train_loader,g_optimizer, d_optimizer, gan_loss, l1_loss,device, args.lambda_l1)
        
        #validation
        val_l1_loss = validate(generator, val_loader, l1_loss, device)
        
        #update learning rates
        g_scheduler.step()
        d_scheduler.step()
        
        #save history
        history['g_loss'].append(g_loss)
        history['d_loss'].append(d_loss)
        history['val_l1_loss'].append(val_l1_loss)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, Val L1: {val_l1_loss:.4f}")
        
        #save sample images
        if (epoch + 1) % args.save_freq == 0:
            save_sample_images(generator, val_loader, device, epoch + 1, output_dir / 'samples')
        
        #save checkpoints
        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_loss': g_loss,
                'd_loss': d_loss,
                'val_l1_loss': val_l1_loss
            }, output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final models
    torch.save(generator.state_dict(), output_dir / 'generator_final.pth')
    torch.save(discriminator.state_dict(), output_dir / 'discriminator_final.pth')
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # # Plot training curves
    # plt.figure(figsize=(15, 5))
    
    # plt.subplot(1, 3, 1)
    # plt.plot(history['g_loss'], label='Generator Loss')
    # plt.plot(history['d_loss'], label='Discriminator Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('GAN Losses')
    # plt.legend()
    # plt.grid(True)
    
    # plt.subplot(1, 3, 2)
    # plt.plot(history['val_l1_loss'], label='Validation L1 Loss', color='red')
    # plt.xlabel('Epoch')
    # plt.ylabel('L1 Loss')
    # plt.title('Validation L1 Loss')
    # plt.legend()
    # plt.grid(True)
    
    # plt.subplot(1, 3, 3)
    # plt.plot(history['g_loss'], label='Generator', alpha=0.7)
    # plt.plot(history['val_l1_loss'], label='Val L1 (scaled)', alpha=0.7)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Overview')
    # plt.legend()
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Plot training curves
    plt.figure(figsize=(15, 5))

    # 1. Generator Loss
    plt.subplot(1, 3, 1)
    plt.plot(history['g_loss'], label='Generator Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()
    plt.grid(True)

    # 2. Discriminator Loss
    plt.subplot(1, 3, 2)
    plt.plot(history['d_loss'], label='Discriminator Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True)

    # 3. Both Losses Together
    plt.subplot(1, 3, 3)
    plt.plot(history['g_loss'], label='Generator Loss', color='blue', alpha=0.7)
    plt.plot(history['d_loss'], label='Discriminator Loss', color='green', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator vs Discriminator Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training completed!")
    print(f"Models saved to: {output_dir}")
    print(f"Sample images saved to: {output_dir / 'samples'}")
    print(f"Training curves saved to: {output_dir / 'training_curves.png'}")

if __name__ == '__main__':
    main()