"""
Worker process for distributed GAN training.
Pulls work units, computes gradients, and uploads to database.
"""

import time
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.dcgan import Generator, Discriminator
from data.dataset import CelebADataset, IndexedDataset, create_dataloader
from database.db_manager import DatabaseManager
from utils import (
    load_config, build_db_url, get_device, get_gpu_name, 
    get_hostname, generate_worker_id, compute_gradient_dict,
    print_training_stats
)


class Worker:
    """Worker for distributed GAN training."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize worker.
        
        Args:
            config_path: Path to configuration file
        """
        print('Initializing worker...')
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup database
        db_url = build_db_url(self.config['database'])
        self.db = DatabaseManager(db_url)
        
        # Generate worker ID
        self.worker_id = generate_worker_id()
        self.hostname = get_hostname()
        self.gpu_name = get_gpu_name()
        
        # Setup device
        self.device = get_device()
        
        # Load dataset
        print('Loading dataset...')
        self.dataset = CelebADataset(
            root_dir=self.config['data']['dataset_path'],
            image_size=self.config['training']['image_size']
        )
        
        # Initialize models
        print('Initializing models...')
        self.generator = Generator(
            latent_dim=self.config['training']['latent_dim']
        ).to(self.device)
        
        self.discriminator = Discriminator().to(self.device)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Register worker in database
        print(f'Registering worker: {self.worker_id}')
        self.db.register_worker(self.worker_id, self.hostname, self.gpu_name)
        
        # Training config
        self.batch_size = self.config['training']['batch_size']
        self.latent_dim = self.config['training']['latent_dim']
        self.poll_interval = self.config['worker']['poll_interval']
        self.heartbeat_interval = self.config['worker']['heartbeat_interval']
        
        # Timing
        self.last_heartbeat = time.time()
        
        print(f'Worker {self.worker_id} initialized successfully!')
        print(f'GPU: {self.gpu_name}')
        print(f'Dataset size: {len(self.dataset)}')
    
    def update_heartbeat(self):
        """Update worker heartbeat in database."""
        current_time = time.time()
        if current_time - self.last_heartbeat > self.heartbeat_interval:
            self.db.update_worker_heartbeat(self.worker_id)
            self.last_heartbeat = current_time
    
    def load_model_weights(self, iteration: int):
        """Load model weights from database.
        
        Args:
            iteration: Training iteration
        """
        # Load generator weights
        gen_weights = self.db.get_model_weights_at_iteration('generator', iteration)
        if gen_weights:
            self.generator.load_state_dict(gen_weights)
        
        # Load discriminator weights
        disc_weights = self.db.get_model_weights_at_iteration('discriminator', iteration)
        if disc_weights:
            self.discriminator.load_state_dict(disc_weights)
    
    def process_work_unit(self, work_unit: Dict):
        """Process a work unit and compute gradients.
        
        Args:
            work_unit: Dictionary with work unit information
        """
        work_unit_id = work_unit['id']
        iteration = work_unit['iteration']
        image_indices = work_unit['image_indices']
        
        print(f'\nProcessing work unit {work_unit_id} (iteration {iteration})')
        print(f'Number of images: {len(image_indices)}')
        
        # Load current model weights
        self.load_model_weights(iteration)
        
        # Create dataloader for this work unit
        indexed_dataset = IndexedDataset(self.dataset, image_indices)
        dataloader = create_dataloader(
            indexed_dataset,
            batch_size=self.batch_size,
            num_workers=self.config['data']['num_workers_dataloader'],
            shuffle=True
        )
        
        # Zero gradients
        self.generator.zero_grad()
        self.discriminator.zero_grad()
        
        # Accumulate gradients over all batches
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_d_real_correct = 0
        total_d_fake_correct = 0
        total_samples = 0
        num_batches = 0
        
        # Labels
        real_label = 1.0
        fake_label = 0.0
        
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            total_samples += batch_size
            
            # ===== Train Discriminator =====
            # Train with real images
            labels = torch.full((batch_size, 1, 1, 1), real_label, device=self.device)
            output_real = self.discriminator(real_images)
            d_loss_real = self.criterion(output_real, labels)
            d_loss_real.backward()
            
            # Calculate accuracy on real images
            d_real_pred = torch.sigmoid(output_real) > 0.5
            total_d_real_correct += d_real_pred.sum().item()
            
            # Train with fake images
            noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            fake_images = self.generator(noise)
            labels.fill_(fake_label)
            output_fake = self.discriminator(fake_images.detach())
            d_loss_fake = self.criterion(output_fake, labels)
            d_loss_fake.backward()
            
            # Calculate accuracy on fake images
            d_fake_pred = torch.sigmoid(output_fake) <= 0.5
            total_d_fake_correct += d_fake_pred.sum().item()
            
            d_loss = d_loss_real + d_loss_fake
            total_d_loss += d_loss.item()
            
            # ===== Train Generator =====
            labels.fill_(real_label)  # Generator wants discriminator to think fakes are real
            output = self.discriminator(fake_images)
            g_loss = self.criterion(output, labels)
            g_loss.backward()
            
            total_g_loss += g_loss.item()
            num_batches += 1
        
        # Average losses
        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        d_real_acc = total_d_real_correct / total_samples
        d_fake_acc = total_d_fake_correct / total_samples
        
        print(f'Completed {num_batches} batches ({total_samples} images)')
        print(f'G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f} | '
              f'D_real: {d_real_acc:.2%} | D_fake: {d_fake_acc:.2%}')
        
        # Extract gradients
        print('Extracting gradients...')
        gen_gradients = compute_gradient_dict(self.generator)
        disc_gradients = compute_gradient_dict(self.discriminator)
        
        # Upload gradients to database
        print('Uploading gradients...')
        self.db.save_gradients(
            worker_id=self.worker_id,
            model_type='generator',
            iteration=iteration,
            work_unit_id=work_unit_id,
            gradients=gen_gradients,
            num_samples=total_samples
        )
        
        self.db.save_gradients(
            worker_id=self.worker_id,
            model_type='discriminator',
            iteration=iteration,
            work_unit_id=work_unit_id,
            gradients=disc_gradients,
            num_samples=total_samples
        )
        
        # Mark work unit as completed
        self.db.complete_work_unit(work_unit_id)
        
        # Update worker statistics
        self.db.update_worker_stats(
            worker_id=self.worker_id,
            work_units=1,
            batches=num_batches,
            images=total_samples
        )
        
        print(f'Work unit {work_unit_id} completed successfully!')
    
    def run(self):
        """Main worker loop."""
        print(f'\nWorker {self.worker_id} starting main loop...')
        print('Waiting for work units...')
        
        try:
            while True:
                # Update heartbeat
                self.update_heartbeat()
                
                # Check if training is still active
                training_state = self.db.get_training_state()
                if training_state and not training_state['training_active']:
                    print('\nTraining has been stopped by coordinator.')
                    break
                
                # Try to claim a work unit
                work_unit = self.db.claim_work_unit(
                    worker_id=self.worker_id,
                    timeout_seconds=self.config['worker']['work_unit_timeout']
                )
                
                if work_unit:
                    # Process the work unit
                    self.process_work_unit(work_unit)
                else:
                    # No work available, wait before polling again
                    time.sleep(self.poll_interval)
        
        except KeyboardInterrupt:
            print('\n\nWorker interrupted by user.')
        except Exception as e:
            print(f'\n\nError in worker: {e}')
            import traceback
            traceback.print_exc()
        finally:
            print(f'\nWorker {self.worker_id} shutting down...')


def main():
    """Main entry point for worker."""
    parser = argparse.ArgumentParser(description='Distributed GAN Training - Worker')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Create and run worker
    worker = Worker(config_path=args.config)
    worker.run()


if __name__ == '__main__':
    main()
