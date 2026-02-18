# Native Python setup

Direct installation on your system without Docker.

## Advantages

- Complete control over environment
- No Docker overhead
- Faster iteration for development
- Works on any system with Python

## Prerequisites

- Python 3.10 or later
- pip package manager
- NVIDIA GPU with CUDA 11.8 (for GPU support)
  - Or CPU-only mode (no GPU required)
- 10GB free disk space

## Installation steps

### 1. Verify Python version

```bash
python --version
# Should show Python 3.10 or later
```

If you need to install Python:
- **Ubuntu/Debian**: `sudo apt install python3.10 python3.10-venv`
- **macOS**: `brew install python@3.10`
- **Windows**: Download from [python.org](https://www.python.org)

### 2. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
cd GANNs-with-freinds
```

### 3. Create virtual environment

```bash
python -m venv venv
```

Activate the environment:

```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### 4. Install dependencies

**With GPU (CUDA 11.8):**
```bash
pip install -r requirements.txt
```

**CPU-only (no GPU):**
```bash
pip install -r .devcontainer/cpu/requirements.txt
```

This installs:
- PyTorch (with or without CUDA)
- torchvision
- PIL for image processing
- PyYAML for configuration
- psycopg2 for database
- huggingface-hub for model sharing

### 5. Verify installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check GPU availability (if installed with GPU support)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check project imports
python -c "from src.models.dcgan import Generator; print('Models OK')"
```

### 6. Download dataset

```bash
python scripts/download_celeba.py
```

This downloads ~1.4 GB to `data/celeba/`.

### 7. Configure database

```bash
cp config.yaml.template config.yaml
```

Edit `config.yaml` with your database credentials:

```yaml
database:
  host: YOUR_DATABASE_HOST
  port: 5432
  database: distributed_gan
  user: YOUR_USERNAME
  password: YOUR_PASSWORD
```

### 8. Start worker

```bash
python src/worker.py
```

Expected output:
```
Initializing worker...
Using GPU: NVIDIA GeForce RTX 3080
# Or: Using CPU
Worker abc123 initialized successfully!
Dataset size: 202,599 images
Polling for work...
```

## Managing the environment

### Activate environment

Always activate the virtual environment before working:

```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### Deactivate environment

When done:

```bash
deactivate
```

### Update dependencies

If requirements change:

```bash
pip install -r requirements.txt --upgrade
```

### Freeze current environment

To save your exact package versions:

```bash
pip freeze > requirements-freeze.txt
```

## CPU vs GPU

The worker automatically detects your hardware:

**With GPU:**
- Uses CUDA for acceleration
- Default batch size (32)
- Faster training

**With CPU:**
- Falls back to CPU automatically
- Smaller batch size (8)
- Slower but works without GPU

## Troubleshooting

**ImportError: No module named 'torch':**
- Make sure virtual environment is activated
- Reinstall: `pip install -r requirements.txt`

**CUDA out of memory:**
- Reduce batch size in `config.yaml`
- Try CPU-only mode

**SSL certificate errors during download:**
```bash
# Ubuntu/Debian
pip install --upgrade certifi

# Or bypass (not recommended)
export CURL_CA_BUNDLE=""
```

**Database connection fails:**
- Check `config.yaml` credentials
- Verify database is accessible
- Test: `psql -h HOST -U USER -d DATABASE`

**Python version too old:**
- Install Python 3.10 or later
- Use pyenv for version management

## Next steps

- [Student guide](../guides/students.md) - How to contribute as a worker
- [Configuration](../guides/configuration.md) - Customize settings
- [Local training](local-training.md) - Train without database
