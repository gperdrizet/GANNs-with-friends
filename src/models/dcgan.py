"""
DCGAN (Deep Convolutional GAN) architecture for CelebA.
Based on the original DCGAN paper: https://arxiv.org/abs/1511.06434
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN Generator for 64x64 images."""
    
    def __init__(self, latent_dim: int = 100, num_channels: int = 3, feature_maps: int = 64):
        """Initialize generator.
        
        Args:
            latent_dim: Dimension of latent noise vector
            num_channels: Number of output channels (3 for RGB)
            feature_maps: Base number of feature maps (affects model capacity)
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Architecture follows DCGAN guidelines:
        # - Use transposed convolutions for upsampling
        # - BatchNorm in all layers except output
        # - ReLU activation in all layers except output (Tanh)
        
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            # Output: (feature_maps * 8) x 4 x 4
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # Output: (feature_maps * 4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # Output: (feature_maps * 2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # Output: feature_maps x 32 x 32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # Output: num_channels x 64 x 64
            nn.ConvTranspose2d(feature_maps, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights according to DCGAN paper."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        """Generate images from noise.
        
        Args:
            z: Noise tensor of shape (batch_size, latent_dim, 1, 1)
            
        Returns:
            Generated images of shape (batch_size, num_channels, 64, 64)
        """
        return self.main(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator for 64x64 images."""
    
    def __init__(self, num_channels: int = 3, feature_maps: int = 64):
        """Initialize discriminator.
        
        Args:
            num_channels: Number of input channels (3 for RGB)
            feature_maps: Base number of feature maps (affects model capacity)
        """
        super(Discriminator, self).__init__()
        
        # Architecture follows DCGAN guidelines:
        # - Use strided convolutions for downsampling
        # - BatchNorm in all layers except first
        # - LeakyReLU activation in all layers
        
        self.main = nn.Sequential(
            # Input: num_channels x 64 x 64
            # Output: feature_maps x 32 x 32
            nn.Conv2d(num_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (feature_maps * 2) x 16 x 16
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (feature_maps * 4) x 8 x 8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: (feature_maps * 8) x 4 x 4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 1 x 1 x 1
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            # No sigmoid - we'll use BCEWithLogitsLoss
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights according to DCGAN paper."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        """Classify images as real or fake.
        
        Args:
            x: Image tensor of shape (batch_size, num_channels, 64, 64)
            
        Returns:
            Logits of shape (batch_size, 1, 1, 1)
        """
        return self.main(x)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    latent_dim = 100
    gen = Generator(latent_dim=latent_dim).to(device)
    disc = Discriminator().to(device)
    
    print(f'Generator parameters: {count_parameters(gen):,}')
    print(f'Discriminator parameters: {count_parameters(disc):,}')
    
    # Test forward pass
    batch_size = 8
    noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
    fake_images = gen(noise)
    print(f'\nGenerated images shape: {fake_images.shape}')
    
    disc_output = disc(fake_images)
    print(f'Discriminator output shape: {disc_output.shape}')
    
    print("\nModels tested successfully!")
