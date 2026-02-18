# Installation

Choose the installation path that best fits your setup and experience level.

## Comparison of installation paths

| Path | GPU required | Docker required | Installation time | Best for |
|------|--------------|-----------------|-------------------|----------|
| **Google Colab** | No (free GPU provided) | No | 2 minutes | Quick start, no local setup |
| **Dev container** | Recommended | Yes | 10 minutes | Full development |
| **Native Python** | Optional | No | 5 minutes | Direct control |
| **Conda** | Optional | No | 5 minutes | Conda users |

## Prerequisites by path

### All paths require:
- 10GB free disk space (for CelebA dataset)
- Internet connection
- Database credentials from instructor (for distributed training)

### Path-specific requirements:

**Google Colab:**
- Google account
- Web browser

**Dev container:**
- Docker with GPU support
- VS Code with Dev Containers extension
- NVIDIA drivers ≥545 (for GPU support)

**Native Python:**
- Python 3.10 or later
- pip package manager
- NVIDIA drivers (for GPU, optional)

**Conda:**
- Anaconda or Miniconda
- NVIDIA drivers (for GPU, optional)

## CPU vs GPU training

Most paths support both CPU and GPU:

- **Google Colab**: Choose GPU or CPU runtime
- **Dev container**: Choose GPU or CPU configuration when opening
- **Native Python / Conda**: Install GPU or CPU requirements, auto-detects hardware
- **Local training**: Works with either

**GPU training**: Faster, recommended for active participation  
**CPU training**: Works but slower

The worker automatically detects your hardware. If you encounter out-of-memory errors, reduce `batch_size` in config.yaml.

## Quick links

Jump directly to your chosen installation method:

- [Google Colab setup](../setup/google-colab.md)
- [Dev container setup](../setup/dev-container.md)
- [Native Python setup](../setup/native-python.md)
- [Conda setup](../setup/conda.md)
- [Local training (no database)](../setup/local-training.md)

## Verification

After installation, verify your setup:

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test project imports
cd src
python -c "from models.dcgan import Generator, Discriminator; print('Models OK')"
python -c "from database.db_manager import DatabaseManager; print('Database OK')"
```

## Next steps

Once installed, proceed to:
- [Quick start guide](quick-start.md)
- [Configuration guide](../guides/configuration.md)
