"""
Main coordinator process for distributed GAN training.
Creates work units, aggregates gradients, and updates models.
"""

import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim

from models.dcgan import Generator, Discriminator
from data.dataset import CelebADataset, get_dataset_indices
from database.db_manager import DatabaseManager
from utils import (
    load_config, build_db_url, get_device,
    apply_gradients, weighted_average_gradients,
    save_generated_images, print_training_stats,
    push_to_huggingface
)


class MainCoordinator:
    """Main coordinator for distributed GAN training."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize coordinator.
        
        Args:
            config_path: Path to configuration file
        """
        print('Initializing main coordinator...')
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup database
        db_url = build_db_url(self.config['database'])
        self.db = DatabaseManager(db_url)
        
        # Setup device
        self.device = get_device()
        
        # Load dataset to get size
        print('Loading dataset...')
        dataset = CelebADataset(
            root_dir=self.config['data']['dataset_path'],
            image_size=self.config['training']['image_size']
        )
        self.dataset_size = len(dataset)
        print(f'Dataset size: {self.dataset_size}')
        
        # Initialize models
        print('Initializing models...')
        self.generator = Generator(
            latent_dim=self.config['training']['latent_dim']
        ).to(self.device)
        
        self.discriminator = Discriminator().to(self.device)
        
        # Initialize optimizers
        lr = self.config['training'].get('learning_rate', 0.0002)
        beta1 = self.config['training'].get('beta1', 0.5)
        beta2 = self.config['training'].get('beta2', 0.999)
        
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        
        # Training config
        self.batch_size = self.config['training']['batch_size']
        self.batches_per_work_unit = self.config['training']['batches_per_work_unit']
        self.min_workers_per_update = self.config['training']['num_workers_per_update']
        self.latent_dim = self.config['training']['latent_dim']
        
        # Calculate work units
        self.images_per_work_unit = self.batch_size * self.batches_per_work_unit
        
        # Fixed noise for generating samples
        self.fixed_noise = torch.randn(64, self.latent_dim, 1, 1, device=self.device)
        
        # Output directories
        self.output_dir = Path('outputs')
        self.samples_dir = self.output_dir / 'samples'
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Hugging Face configuration
        self.hf_enabled = self.config.get('huggingface', {}).get('enabled', False)
        self.hf_repo_id = self.config.get('huggingface', {}).get('repo_id', '')
        self.hf_token = self.config.get('huggingface', {}).get('token', '')
        self.hf_push_interval = self.config.get('huggingface', {}).get('push_interval', 5)
        
        if self.hf_enabled:
            print(f'Hugging Face Hub integration enabled: {self.hf_repo_id}')
        
        print('Main coordinator initialized successfully!')
    
    def initialize_training(self):
        """Initialize training state in database."""
        print('\nInitializing training state...')
        
        # Save initial model weights
        self.db.save_model_weights('generator', 0, self.generator.state_dict())
        self.db.save_model_weights('discriminator', 0, self.discriminator.state_dict())
        
        # Save initial optimizer states
        self.db.save_optimizer_state('generator', 0, self.optimizer_g.state_dict())
        self.db.save_optimizer_state('discriminator', 0, self.optimizer_d.state_dict())
        
        # Reset training state
        self.db.update_training_state(
            current_iteration=0,
            current_epoch=0,
            total_batches_processed=0,
            total_images_processed=0,
            training_active=True
        )
        
        print('Training state initialized!')
    
    def create_work_units_for_iteration(self, iteration: int):
        """Create work units for current iteration.
        
        Args:
            iteration: Current training iteration
        """
        # Split dataset into work units
        work_units_indices = get_dataset_indices(
            self.dataset_size,
            self.batch_size,
            self.batches_per_work_unit
        )
        
        # Create work units in database
        work_unit_ids = self.db.create_work_units(
            iteration=iteration,
            image_indices_list=work_units_indices,
            num_batches_per_unit=self.batches_per_work_unit,
            timeout_seconds=self.config['worker']['work_unit_timeout']
        )
        
        print(f'Created {len(work_unit_ids)} work units for iteration {iteration}')
        return len(work_unit_ids)
    
    def wait_for_gradients(self, iteration: int, total_work_units: int):
        """Wait for gradients from workers.
        
        Args:
            iteration: Current training iteration
            total_work_units: Total number of work units
            
        Returns:
            True if enough gradients collected, False if should stop
        """
        print(f'\nWaiting for workers to compute gradients...')
        
        last_update = time.time()
        update_interval = 10  # Print status every 10 seconds
        
        while True:
            # Check work unit completion status
            stats = self.db.get_work_unit_stats(iteration)
            completed = stats['completed']
            total = stats['total']
            
            # Print status update
            current_time = time.time()
            if current_time - last_update > update_interval:
                active_workers = self.db.get_active_workers()
                print(f'Progress: {completed}/{total} work units completed | '
                      f'Active workers: {len(active_workers)}')
                last_update = current_time
            
            # Check if all work units are completed
            if completed >= total:
                print(f'All work units completed for iteration {iteration}')
                return True
            
            # Check if we have minimum number of gradients
            gen_gradients = self.db.get_gradients_for_iteration('generator', iteration)
            if len(gen_gradients) >= self.min_workers_per_update:
                print(f'Minimum threshold reached: {len(gen_gradients)} gradients collected')
                return True
            
            # Sleep before checking again
            time.sleep(2)
    
    def aggregate_and_update(self, iteration: int):
        """Aggregate gradients and update models.
        
        Args:
            iteration: Current training iteration
        """
        print('\nAggregating gradients...')
        
        # Get gradients from database
        gen_gradients_info = self.db.get_gradients_for_iteration('generator', iteration)
        disc_gradients_info = self.db.get_gradients_for_iteration('discriminator', iteration)
        
        num_workers = len(gen_gradients_info)
        print(f'Received gradients from {num_workers} workers')
        
        if num_workers == 0:
            print('No gradients to aggregate!')
            return
        
        # Average gradients (weighted by number of samples)
        avg_gen_gradients = weighted_average_gradients(gen_gradients_info)
        avg_disc_gradients = weighted_average_gradients(disc_gradients_info)
        
        # Apply gradients to models
        apply_gradients(self.generator, avg_gen_gradients)
        apply_gradients(self.discriminator, avg_disc_gradients)
        
        # Optimizer step
        print('Applying optimizer step...')
        self.optimizer_g.step()
        self.optimizer_d.step()
        
        # Zero gradients
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        
        # Save updated weights and optimizer states
        next_iteration = iteration + 1
        self.db.save_model_weights('generator', next_iteration, self.generator.state_dict())
        self.db.save_model_weights('discriminator', next_iteration, self.discriminator.state_dict())
        self.db.save_optimizer_state('generator', next_iteration, self.optimizer_g.state_dict())
        self.db.save_optimizer_state('discriminator', next_iteration, self.optimizer_d.state_dict())
        
        # Clean up gradients from database
        self.db.delete_gradients_for_iteration(iteration)
        
        print('Models updated successfully!')
    
    def generate_samples(self, iteration: int):
        """Generate and save sample images.
        
        Args:
            iteration: Current training iteration
        """
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
        self.generator.train()
        
        output_path = self.samples_dir / f'iteration_{iteration:06d}.png'
        save_generated_images(fake_images, str(output_path))
        print(f'Saved sample images to {output_path}')
        
        return output_path
    
    def push_to_hub(self, iteration: int, epoch: int, samples_path: str = None):
        """Push model to Hugging Face Hub if enabled.
        
        Args:
            iteration: Current training iteration
            epoch: Current epoch
            samples_path: Optional path to sample images
        """
        if self.hf_enabled and self.hf_repo_id and self.hf_token:
            if iteration % self.hf_push_interval == 0:
                push_to_huggingface(
                    generator=self.generator,
                    discriminator=self.discriminator,
                    iteration=iteration,
                    epoch=epoch,
                    repo_id=self.hf_repo_id,
                    token=self.hf_token,
                    samples_path=samples_path
                )
    
    def run(self, num_epochs: int = 50, sample_interval: int = 100):
        """Run main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            sample_interval: Generate samples every N iterations
        """
        print('\n' + '='*70)
        print('Starting Distributed GAN Training')
        print('='*70)
        
        # Initialize training
        self.initialize_training()
        
        # Calculate total iterations per epoch
        images_per_work_unit = self.batch_size * self.batches_per_work_unit
        work_units_per_epoch = self.dataset_size // images_per_work_unit
        
        print(f'\nTraining configuration:')
        print(f'  Epochs: {num_epochs}')
        print(f'  Dataset size: {self.dataset_size}')
        print(f'  Batch size: {self.batch_size}')
        print(f'  Batches per work unit: {self.batches_per_work_unit}')
        print(f'  Images per work unit: {images_per_work_unit}')
        print(f'  Work units per epoch: {work_units_per_epoch}')
        print(f'  Min workers per update: {self.min_workers_per_update}')
        
        iteration = 0
        
        try:
            for epoch in range(num_epochs):
                print(f"\n{'='*70}")
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print('='*70)
                
                # Create work units for this iteration
                total_work_units = self.create_work_units_for_iteration(iteration)
                
                # Wait for workers to complete work units
                if not self.wait_for_gradients(iteration, total_work_units):
                    print('Training stopped.')
                    break
                
                # Aggregate gradients and update models
                self.aggregate_and_update(iteration)
                
                # Update training state
                self.db.update_training_state(
                    current_iteration=iteration + 1,
                    current_epoch=epoch
                )
                
                # Generate sample images periodically
                if iteration % sample_interval == 0:
                    print('\nGenerating sample images...')
                    samples_path = self.generate_samples(iteration)
                    
                    # Push to Hugging Face Hub if enabled
                    self.push_to_hub(iteration, epoch, str(samples_path))
                
                # Print active workers
                active_workers = self.db.get_active_workers()
                print(f'\nActive workers: {len(active_workers)}')
                for worker in active_workers[:5]:  # Show first 5
                    print(f"  - {worker['worker_id']}: {worker['total_work_units']} work units, "
                          f"{worker['total_images']} images")
                
                iteration += 1
        
        except KeyboardInterrupt:
            print('\n\nTraining interrupted by user.')
        except Exception as e:
            print(f'\n\nError in training: {e}')
            import traceback
            traceback.print_exc()
        finally:
            # Mark training as inactive
            self.db.update_training_state(training_active=False)
            print('\n\nTraining stopped. Workers will stop automatically.')
            
            # Generate final samples
            print('Generating final samples...')
            self.generate_samples(iteration)
            
            # Print final statistics
            print('\n' + '='*70)
            print('Training Complete!')
            print('='*70)
            print(f'Total iterations: {iteration}')
            active_workers = self.db.get_active_workers(timeout_seconds=3600)
            print(f'Total workers participated: {len(active_workers)}')
            total_images = sum(w['total_images'] for w in active_workers)
            print(f'Total images processed: {total_images}')


def main():
    """Main entry point for coordinator."""
    parser = argparse.ArgumentParser(description='Distributed GAN Training - Main Coordinator')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--sample-interval',
        type=int,
        default=1,
        help='Generate samples every N iterations'
    )
    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize training state (use for first run)'
    )
    
    args = parser.parse_args()
    
    # Create coordinator
    coordinator = MainCoordinator(config_path=args.config)
    
    # Run training
    coordinator.run(num_epochs=args.epochs, sample_interval=args.sample_interval)


if __name__ == '__main__':
    main()
