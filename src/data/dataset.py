"""
Dataset loader for CelebA and other image datasets.
"""

import os
import io
import zipfile
from typing import List, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


class CelebADataset(Dataset):
    """CelebA dataset loader - supports both extracted folders and zip files."""
    
    def __init__(
        self, 
        root_dir: str,
        image_size: int = 64,
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize CelebA dataset.
        
        Args:
            root_dir: Root directory containing CelebA images, OR path to zip file
            image_size: Target image size (images will be resized)
            transform: Optional transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.zip_file = None
        self.zip_path = None
        
        # Check if we should load from a zip file
        # Look for zip file in multiple possible locations
        zip_locations = [
            # Direct parent (e.g., data/celeba_torchvision/data/img_align_celeba.zip)
            self.root_dir.parent / 'img_align_celeba.zip',
            # HF download location: {base}/data/img_align_celeba.zip
            self.root_dir.parent.parent / 'data' / 'img_align_celeba.zip',
        ]
        
        zip_found = None
        for possible_zip in zip_locations:
            if possible_zip.exists():
                zip_found = possible_zip
                break
        
        if zip_found and not self.root_dir.exists():
            self._init_from_zip(zip_found)
        elif str(root_dir).endswith('.zip') and Path(root_dir).exists():
            self._init_from_zip(Path(root_dir))
        else:
            self._init_from_folder()
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
        
        print(f'Loaded dataset with {len(self.image_files)} images')
    
    def _init_from_folder(self):
        """Initialize from extracted folder."""
        self.loading_from_zip = False
        self._zip_handle = None  # Not used for folder loading
        
        # Find all image files
        self.image_files = sorted([
            f for f in self.root_dir.glob('*.jpg') 
            if f.is_file()
        ])
        
        # Also check for PNG files
        if not self.image_files:
            self.image_files = sorted([
                f for f in self.root_dir.glob('*.png') 
                if f.is_file()
            ])
        
        if not self.image_files:
            raise ValueError(f'No image files found in {self.root_dir}')
    
    def _init_from_zip(self, zip_path: Path):
        """Initialize from zip file without extracting."""
        self.loading_from_zip = True
        self.zip_path = str(zip_path)  # Store as string for pickling
        self._zip_handle = None  # Lazy-loaded, per-process
        
        print(f'Loading dataset directly from zip: {zip_path}')
        print('(No extraction needed - this saves disk I/O)')
        
        # Open zip temporarily just to get the file list
        with zipfile.ZipFile(zip_path, 'r') as zf:
            self.image_files = sorted([
                name for name in zf.namelist()
                if name.endswith('.jpg') or name.endswith('.png')
            ])
        
        if not self.image_files:
            raise ValueError(f'No image files found in {zip_path}')
    
    def _get_zip_handle(self):
        """Get zip file handle, opening lazily for multiprocessing support."""
        if self._zip_handle is None:
            self._zip_handle = zipfile.ZipFile(self.zip_path, 'r')
        return self._zip_handle
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get image at index.
        
        Args:
            idx: Image index
            
        Returns:
            Transformed image tensor and index
        """
        if self.loading_from_zip:
            # Read from zip file (lazy-open for multiprocessing)
            zf = self._get_zip_handle()
            img_data = zf.read(self.image_files[idx])
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
        else:
            # Read from disk
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, idx


class IndexedDataset(Dataset):
    """Dataset that returns images at specific indices."""
    
    def __init__(self, base_dataset: Dataset, indices: List[int]):
        """Initialize indexed dataset.
        
        Args:
            base_dataset: Base dataset to sample from
            indices: List of indices to include
        """
        self.base_dataset = base_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get image at mapped index."""
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True
) -> DataLoader:
    """Create a DataLoader with standard settings.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch
        prefetch_factor=4,        # Pre-load 4 batches per worker for better GPU utilization
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between epochs
    )


def get_dataset_indices(dataset_size: int, images_per_work_unit: int) -> List[List[int]]:
    """Split dataset indices into work units.
    
    Args:
        dataset_size: Total number of images in dataset
        images_per_work_unit: Number of images per work unit
        
    Returns:
        List of lists, where each inner list contains indices for one work unit
    """
    all_indices = list(range(dataset_size))
    
    work_units = []
    for i in range(0, len(all_indices), images_per_work_unit):
        work_unit_indices = all_indices[i:i + images_per_work_unit]
        # Only include complete work units
        if len(work_unit_indices) == images_per_work_unit:
            work_units.append(work_unit_indices)
    
    return work_units


if __name__ == '__main__':
    # Test the dataset loader
    import sys
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = './data/celeba'
    
    print(f'Loading dataset from {dataset_path}')
    
    try:
        dataset = CelebADataset(dataset_path, image_size=64)
        print(f'Dataset size: {len(dataset)}')
        
        # Test loading a sample
        img, idx = dataset[0]
        print(f'Image shape: {img.shape}')
        print(f'Image range: [{img.min():.2f}, {img.max():.2f}]')
        
        # Test dataloader
        dataloader = create_dataloader(dataset, batch_size=32, num_workers=0)
        images, indices = next(iter(dataloader))
        print(f'Batch shape: {images.shape}')
        
        # Test work unit splitting
        work_units = get_dataset_indices(len(dataset), images_per_work_unit=320)
        print(f'Number of work units: {len(work_units)}')
        print(f'Images per work unit: {len(work_units[0])}')
        
        print('\nDataset loader tested successfully!')
    except Exception as e:
        print(f'Error: {e}')
        print('\nTo test with actual data, run:')
        print('  python dataset.py /path/to/celeba/images')
