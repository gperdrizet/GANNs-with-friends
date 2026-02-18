"""
Local single-GPU training for DCGAN.
Standard PyTorch training loop without distributed coordination.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.dcgan import Generator, Discriminator
from data.dataset import CelebADataset
from utils import (
    load_config, get_device, save_generated_images, 
    print_training_stats, format_time
)


class LocalTrainer:
    """Local single-GPU trainer for DCGAN."""
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = 'outputs_local',
        batch_size: int = 128,
        latent_dim: int = 100,
        image_size: int = 64,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        num_workers: int = 4,
        gpu_id: int = 0
    ):
        """Initialize local trainer.
        
        Args:
            dataset_path: Path to image dataset
            output_dir: Output directory for samples and checkpoints
            batch_size: Batch size for training
            latent_dim: Latent dimension for generator
            image_size: Image size (height and width)
            lr: Learning rate
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
            num_workers: Number of dataloader workers
            gpu_id: GPU device ID to use (default: 0)
        """
        self.device = get_device(gpu_id)
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
        # Create output directories
        self.output_dir = Path(output_dir)
        self.samples_dir = self.output_dir / 'samples'
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print(f'Loading dataset from {dataset_path}...')
        dataset = CelebADataset(dataset_path, image_size=image_size)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,        # Pre-load 4 batches per worker for better GPU utilization
            persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
        )
        print(f'Dataset loaded: {len(dataset)} images')
        
        # Initialize models
        print('Initializing models...')
        self.generator = Generator(latent_dim=latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Fixed noise for consistent sample generation
        self.fixed_noise = torch.randn(64, latent_dim, 1, 1, device=self.device)
        
        # Training statistics
        self.g_losses = []
        self.d_losses = []
        
        print('Local trainer initialized successfully!')
        print(f'Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}')
        print(f'Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}')
    
    def train_epoch(self, epoch: int):
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average G loss, average D loss)
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real_correct = 0
        epoch_d_fake_correct = 0
        total_samples = 0
        
        # Labels
        real_label = 1.0
        fake_label = 0.0
        
        start_time = time.time()
        
        for batch_idx, (real_images, _) in enumerate(self.dataloader):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            total_samples += batch_size
            
            # ==================== Train Discriminator ====================
            self.discriminator.zero_grad()
            
            # Train with real images
            labels = torch.full((batch_size, 1, 1, 1), real_label, device=self.device)
            output_real = self.discriminator(real_images)
            d_loss_real = self.criterion(output_real, labels)
            d_loss_real.backward()
            
            # Calculate accuracy on real images
            d_real_pred = torch.sigmoid(output_real) > 0.5
            epoch_d_real_correct += d_real_pred.sum().item()
            
            # Train with fake images
            noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            fake_images = self.generator(noise)
            labels.fill_(fake_label)
            output_fake = self.discriminator(fake_images.detach())
            d_loss_fake = self.criterion(output_fake, labels)
            d_loss_fake.backward()
            
            # Calculate accuracy on fake images
            d_fake_pred = torch.sigmoid(output_fake) <= 0.5
            epoch_d_fake_correct += d_fake_pred.sum().item()
            
            # Update discriminator
            d_loss = d_loss_real + d_loss_fake
            self.optimizer_d.step()
            
            epoch_d_loss += d_loss.item()
            
            # ==================== Train Generator ====================
            self.generator.zero_grad()
            
            # Generator wants discriminator to think fakes are real
            labels.fill_(real_label)
            output = self.discriminator(fake_images)
            g_loss = self.criterion(output, labels)
            g_loss.backward()
            
            # Update generator
            self.optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            
            # Print progress
            if (batch_idx + 1) % 50 == 0:
                avg_g_loss = epoch_g_loss / (batch_idx + 1)
                avg_d_loss = epoch_d_loss / (batch_idx + 1)
                d_real_acc = epoch_d_real_correct / total_samples
                d_fake_acc = epoch_d_fake_correct / total_samples
                
                print(f'  [{batch_idx + 1}/{len(self.dataloader)}] '
                      f'G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f} | '
                      f'D_real: {d_real_acc:.2%} | D_fake: {d_fake_acc:.2%}')
        
        # Calculate epoch averages
        num_batches = len(self.dataloader)
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        d_real_acc = epoch_d_real_correct / total_samples
        d_fake_acc = epoch_d_fake_correct / total_samples
        
        epoch_time = time.time() - start_time
        
        print(f'\nEpoch {epoch} completed in {format_time(epoch_time)}')
        print_training_stats(
            iteration=epoch,
            epoch=epoch,
            g_loss=avg_g_loss,
            d_loss=avg_d_loss,
            d_real_acc=d_real_acc,
            d_fake_acc=d_fake_acc
        )
        
        self.g_losses.append(avg_g_loss)
        self.d_losses.append(avg_d_loss)
        
        return avg_g_loss, avg_d_loss
    
    def generate_samples(self, epoch: int):
        """Generate and save sample images.
        
        Args:
            epoch: Current epoch number
        """
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
        self.generator.train()
        
        output_path = self.samples_dir / f'epoch_{epoch:04d}.png'
        save_generated_images(fake_images, str(output_path))
        print(f'Saved samples to {output_path}')
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
        }
        
        checkpoint_path = self.checkpoints_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = self.checkpoints_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        print(f'Saved checkpoint to {checkpoint_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Starting epoch number
        """
        print(f'Loading checkpoint from {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.g_losses = checkpoint.get('g_losses', [])
        self.d_losses = checkpoint.get('d_losses', [])
        
        epoch = checkpoint['epoch']
        print(f'Loaded checkpoint from epoch {epoch}')
        return epoch + 1
    
    def train(self, num_epochs: int, sample_interval: int = 1, checkpoint_interval: int = 5, resume: str = None):
        """Run training loop.
        
        Args:
            num_epochs: Number of epochs to train
            sample_interval: Generate samples every N epochs
            checkpoint_interval: Save checkpoint every N epochs
            resume: Path to checkpoint to resume from
        """
        start_epoch = 0
        
        if resume:
            start_epoch = self.load_checkpoint(resume)
        
        print('\n' + '='*70)
        print('Starting Local DCGAN Training')
        print('='*70)
        print(f'Epochs: {num_epochs}')
        print(f'Batch size: {self.batch_size}')
        print(f'Batches per epoch: {len(self.dataloader)}')
        print(f'Total images: {len(self.dataloader.dataset)}')
        print(f'Device: {self.device}')
        print('='*70)
        
        try:
            for epoch in range(start_epoch, num_epochs):
                print(f"\n{'='*70}")
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print('='*70)
                
                # Train for one epoch
                g_loss, d_loss = self.train_epoch(epoch + 1)
                
                # Generate samples
                if (epoch + 1) % sample_interval == 0:
                    self.generate_samples(epoch + 1)
                
                # Save checkpoint
                if (epoch + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(epoch + 1)
            
            # Final checkpoint and samples
            print('\nTraining complete!')
            self.save_checkpoint(num_epochs)
            self.generate_samples(num_epochs)
            
        except KeyboardInterrupt:
            print('\n\nTraining interrupted by user.')
            print('Saving checkpoint...')
            self.save_checkpoint(epoch + 1)
            self.generate_samples(epoch + 1)
        
        print('\n' + '='*70)
        print('Training Complete!')
        print('='*70)
        print(f'Final G loss: {self.g_losses[-1]:.4f}')
        print(f'Final D loss: {self.d_losses[-1]:.4f}')
        print(f'Samples saved to: {self.samples_dir}')
        print(f'Checkpoints saved to: {self.checkpoints_dir}')


def main():
    """Main entry point."""
    # Load config first to use as defaults
    config = load_config('config.yaml')
    
    parser = argparse.ArgumentParser(description='Local DCGAN Training')
    
    # Data arguments
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=config['data']['dataset_path'],
        help='Path to dataset directory'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config['training']['batch_size'],
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=config['training'].get('learning_rate', 0.0002),
        help='Learning rate'
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=config['training'].get('beta1', 0.5),
        help='Adam beta1 parameter'
    )
    parser.add_argument(
        '--beta2',
        type=float,
        default=config['training'].get('beta2', 0.999),
        help='Adam beta2 parameter'
    )
    
    # Model arguments
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=config['training']['latent_dim'],
        help='Latent dimension'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=config['training']['image_size'],
        help='Image size'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs_local',
        help='Output directory'
    )
    parser.add_argument(
        '--sample-interval',
        type=int,
        default=1,
        help='Generate samples every N epochs'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=5,
        help='Save checkpoint every N epochs'
    )
    
    # Misc arguments
    parser.add_argument(
        '--num-workers',
        type=int,
        default=config['data'].get('num_workers_dataloader', 4),
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = LocalTrainer(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        num_workers=args.num_workers,
        gpu_id=args.gpu
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        sample_interval=args.sample_interval,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
