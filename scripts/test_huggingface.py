"""
Test Hugging Face Hub integration.

This script verifies that:
1. huggingface_hub is installed
2. Authentication works
3. Model can be pushed and downloaded
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, '../src')

import torch
from models.dcgan import Generator, Discriminator


def test_huggingface_installation():
    """Test if huggingface_hub is installed."""
    print('Testing Hugging Face Hub installation...')
    try:
        import huggingface_hub
        print(f'\u2713 huggingface_hub version: {huggingface_hub.__version__}')
        return True
    except ImportError:
        print('\u2717 huggingface_hub not installed')
        print('Install with: pip install huggingface-hub')
        return False


def test_authentication(token: str):
    """Test if authentication works."""
    print('\nTesting authentication...')
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f'\u2713 Authenticated as: {user_info["name"]}')
        return True
    except Exception as e:
        print(f'\u2717 Authentication failed: {e}')
        return False


def test_push_and_download(repo_id: str, token: str):
    """Test pushing and downloading a model."""
    print('\nTesting model push and download...')
    try:
        from huggingface_hub import HfApi, create_repo, hf_hub_download
        import tempfile
        import json
        
        # Create test models
        generator = Generator(latent_dim=100)
        discriminator = Discriminator()
        
        # Initialize API
        api = HfApi(token=token)
        
        # Create repo
        print(f'Creating repository: {repo_id}')
        try:
            create_repo(repo_id, token=token, exist_ok=True, private=False)
            print(f'\u2713 Repository created/verified')
        except Exception as e:
            print(f'Note: {e}')
        
        # Push checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint.pth'
            
            checkpoint = {
                'iteration': 0,
                'epoch': 0,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            
            print('Pushing checkpoint...')
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo='checkpoint_test.pth',
                repo_id=repo_id,
                repo_type='model',
                commit_message='Test checkpoint'
            )
            print('\u2713 Checkpoint pushed successfully')
            
            # Push metadata
            metadata_path = Path(tmpdir) / 'metadata.json'
            metadata = {
                'test': True,
                'model_type': 'DCGAN',
                'iteration': 0
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            api.upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo='metadata_test.json',
                repo_id=repo_id,
                repo_type='model'
            )
            print('\u2713 Metadata pushed successfully')
        
        # Download checkpoint
        print('Downloading checkpoint...')
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename='checkpoint_test.pth',
            repo_type='model'
        )
        
        # Verify checkpoint
        loaded_checkpoint = torch.load(downloaded_path, map_location='cpu')
        assert 'generator_state_dict' in loaded_checkpoint
        assert 'discriminator_state_dict' in loaded_checkpoint
        print('\u2713 Checkpoint downloaded and verified')
        
        # Download metadata
        metadata_path = hf_hub_download(
            repo_id=repo_id,
            filename='metadata_test.json',
            repo_type='model'
        )
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata['test'] == True
        print('\u2713 Metadata downloaded and verified')
        
        print(f'\n\u2713 All tests passed!')
        print(f'View your test model at: https://huggingface.co/{repo_id}')
        
        return True
        
    except Exception as e:
        print(f'\u2717 Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print('='*70)
    print('Hugging Face Hub Integration Test')
    print('='*70)
    
    # Check installation
    if not test_huggingface_installation():
        return
    
    # Get credentials
    print('\n' + '='*70)
    print('Please provide your Hugging Face credentials:')
    print('Get your token from: https://huggingface.co/settings/tokens')
    print('='*70)
    
    repo_id = input('Repository ID (e.g., username/model-name): ').strip()
    token = input('Access token: ').strip()
    
    if not repo_id or not token:
        print('Error: Repository ID and token are required')
        return
    
    # Run tests
    if test_authentication(token):
        test_push_and_download(repo_id, token)
    
    print('\n' + '='*70)
    print('Test complete!')
    print('='*70)


if __name__ == '__main__':
    main()
