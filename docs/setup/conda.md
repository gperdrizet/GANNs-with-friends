# Conda environment setup

Installation using Anaconda or Miniconda package manager.

## Advantages

- Simplified dependency management
- Easy environment switching
- Cross-platform consistency
- Integrated package solver

## Prerequisites

- Anaconda or Miniconda installed
- NVIDIA GPU with drivers (for GPU support)
  - Or CPU-only mode (no GPU required)
- 10GB free disk space

## Installation steps

### 1. Install Conda

If you don't have conda installed:

**Miniconda (lightweight):**
```bash
# Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS
brew install miniconda
```

**Anaconda (full distribution):**
- Download from [anaconda.com](https://www.anaconda.com/download)

Verify installation:
```bash
conda --version
```

### 2. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
cd GANNs-with-freinds
```

### 3. Create conda environment

**With GPU:**
```bash
conda env create -f environment.yml
```

**CPU-only (no GPU):**
```bash
conda env create -f environment-cpu.yml
```

This creates an environment named `ganns-with-friends` (or `ganns-with-friends-cpu`) with all dependencies.

### 4. Activate environment

```bash
# GPU environment
conda activate ganns-with-friends

# CPU environment
conda activate ganns-with-friends-cpu
```

You should see the environment name in your prompt.

### 5. Verify installation

```bash
# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check GPU (if using GPU environment)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test imports
python -c "from src.models.dcgan import Generator; print('Models OK')"
```

### 6. Download dataset

```bash
python scripts/download_celeba.py
```

Downloads ~1.4 GB to `data/celeba/`.

### 7. Configure database

```bash
cp config.yaml.template config.yaml
```

Edit `config.yaml`:

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
Worker abc123 initialized successfully!
Polling for work...
```

## Managing conda environments

### List environments

```bash
conda env list
```

### Activate/deactivate

```bash
# Activate
conda activate ganns-with-friends

# Deactivate
conda deactivate
```

### Update environment

If `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

### Export environment

Save your current environment:

```bash
conda env export > environment-freeze.yml
```

### Remove environment

If you need to start fresh:

```bash
conda env remove -n ganns-with-friends
```

## Environment files explained

### environment.yml (GPU)

Includes:
- Python 3.10
- PyTorch with CUDA 11.8
- torchvision with GPU support
- PIL, PyYAML, psycopg2
- huggingface-hub (via pip)

### environment-cpu.yml (CPU-only)

Same as above but:
- PyTorch CPU-only build
- Smaller download size
- No CUDA dependencies

## Troubleshooting

**Solving environment takes forever:**
- Try mamba (faster solver): `conda install mamba`
- Then use: `mamba env create -f environment.yml`

**Conflicts with existing packages:**
- Use a fresh environment
- Don't mix pip and conda for same packages

**CUDA version mismatch:**
- Check your driver: `nvidia-smi`
- Adjust CUDA version in `environment.yml`
- Available versions: 11.6, 11.7, 11.8, 12.1

**Database module errors:**
- Install separately: `conda install psycopg2`
- Or use binary: `pip install psycopg2-binary`

**Environment activation fails:**
- Initialize conda: `conda init bash` (or your shell)
- Restart shell
- Try again

## Next steps

- [Student guide](../guides/students.md) - Participate as a worker
- [Configuration](../guides/configuration.md) - Customize your setup  
- [Local training](local-training.md) - Train independently
