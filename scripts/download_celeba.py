"""
Script to download and preprocess CelebA dataset.
"""

import os
import sys
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_celeba(output_dir='./data'):
    """Download CelebA dataset from Google Drive.
    
    Note: This uses the aligned & cropped version of CelebA.
    If this link doesn't work, you may need to download manually from:
    https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    or
    https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('='*70)
    print('CelebA Dataset Download')
    print('='*70)
    print('\nNOTE: Due to Google Drive download restrictions, you may need to')
    print('download CelebA manually. Here are the recommended sources:')
    print('\n1. Kaggle: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset')
    print('2. Official: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html')
    print('\nAfter downloading, extract the images to: ./data/celeba/')
    print('\nAlternatively, you can use torchvision to download automatically:')
    print('See the code below (requires torchvision).\n')
    
    # Try using torchvision
    try:
        from torchvision.datasets import CelebA
        import torchvision.transforms as transforms
        
        print('Attempting to download using torchvision...')
        print('This may take a while (the dataset is ~1.4 GB)')
        
        celeba_dir = output_dir / 'celeba_torchvision'
        
        # Download dataset
        dataset = CelebA(
            root=str(celeba_dir),
            split='all',  # Download all images
            download=True
        )
        
        print(f'\nDataset downloaded successfully!')
        print(f"Location: {celeba_dir / 'celeba'}")
        print(f'Number of images: {len(dataset)}')
        print('\nYou can now add this to your config.yaml:')
        print(f"  dataset_path: {celeba_dir / 'celeba' / 'img_align_celeba'}")
        
        return True
        
    except ImportError:
        print('\ntorchvision not available for automatic download.')
        print('Please download manually as described above.')
        return False
    except Exception as e:
        print(f'\nError during download: {e}')
        print('Please download manually as described above.')
        return False


def verify_dataset(dataset_path):
    """Verify that the dataset is properly downloaded.
    
    Args:
        dataset_path: Path to dataset directory
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f'Error: Dataset path does not exist: {dataset_path}')
        return False
    
    # Count image files
    image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
    num_images = len(image_files)
    
    if num_images == 0:
        print(f'Error: No images found in {dataset_path}')
        return False
    
    print(f'\nDataset verification:')
    print(f'  Path: {dataset_path}')
    print(f'  Number of images: {num_images}')
    
    if num_images < 100:
        print(f'  Warning: Very small dataset ({num_images} images)')
        print(f'  CelebA should have ~200,000 images')
        return False
    
    print(f'  ✓ Dataset looks good!')
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download CelebA dataset')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--verify',
        type=str,
        default=None,
        help='Verify existing dataset at specified path'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset(args.verify)
    else:
        download_celeba(args.output_dir)


if __name__ == '__main__':
    main()
