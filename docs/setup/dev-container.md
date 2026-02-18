# Dev container setup

Full development environment using VS Code dev containers with Docker.

**Two configurations available:**
- **GPU**: For Linux/WSL2 with NVIDIA GPU (requires GPU Docker support)
- **CPU**: For any machine including Macs (no GPU required)

Choose the configuration that matches your hardware when VS Code prompts you.

## GPU configuration

### Advantages

- Pre-configured development environment
- GPU acceleration for faster training
- Isolated from host system
- Reproducible setup
- Full IDE features

### Prerequisites

- Docker Desktop with GPU support (Linux or WSL2 required)
- NVIDIA drivers (version ≥545)
- VS Code with Dev Containers extension
- 15GB free disk space

## CPU configuration

### Advantages

- Works on any machine (Mac, Windows, Linux)
- No GPU required
- Same development environment as GPU
- Good for code development and testing

### Prerequisites

- Docker Desktop (no GPU support needed)
- VS Code with Dev Containers extension
- 10GB free disk space

## Installation steps

### 1. Install prerequisites

**Docker Desktop:**
- Download from [docker.com](https://www.docker.com/products/docker-desktop)
- Enable GPU support in settings

**NVIDIA drivers:**
```bash
# Check current version
nvidia-smi

# Should show driver version ≥545
```

**VS Code extension:**
- Open VS Code
- Install "Dev Containers" extension by Microsoft

### 2. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
cd GANNs-with-freinds
```

### 3. Open in container

1. Open the project folder in VS Code
2. VS Code will detect multiple `.devcontainer` configurations
3. **Choose your configuration:**
   - **DeepLearning GPU**: If you have NVIDIA GPU and Docker GPU support
   - **DeepLearning CPU**: If you're on Mac or don't have GPU
4. Click "Reopen in Container"
   - Or press `F1` → "Dev Containers: Reopen in Container"
5. Wait for container to build (5-10 minutes first time)

**GPU container includes:**
- Python 3.10
- PyTorch with CUDA support
- All project dependencies pre-installed
- Jupyter notebook support

**CPU container includes:**
- Python 3.10
- PyTorch CPU-only (installed during postCreateCommand)
- All project dependencies
- Jupyter notebook support

### 4. Download dataset

Once the container is running:

```bash
python scripts/download_celeba.py
```

This downloads ~1.4 GB of celebrity face images.

### 5. Configure database

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

### 6. Start worker

```bash
python src/worker.py
```

**GPU configuration output:**
```
Initializing worker...
Using GPU: NVIDIA GeForce RTX 3080
Worker abc123 initialized successfully!
Polling for work...
```

**CPU configuration output:**
```
Initializing worker...
Using CPU
Worker abc123 initialized successfully!
Polling for work...
```

## Container features

### Hardware detection

The worker automatically detects your hardware:

**GPU configuration:**
```bash
# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
python -c "import torch; print(torch.cuda.get_device_name(0))"  # Shows GPU name
```

**CPU configuration:**
```bash
# Verify CPU mode
python -c "import torch; print(torch.cuda.is_available())"  # Prints False
# Batch size will be automatically reduced for CPU training
```

### Jupyter notebooks

Access the demo notebook:

```bash
jupyter notebook notebooks/demo_trained_model.ipynb
```

The container forwards port 8888 to your host.

### Development tools

Pre-installed tools:
- Git
- pytest for testing
- pylint for linting
- black for code formatting

## Working with the container

### Restart container

If you need to restart:
1. `F1` → "Dev Containers: Rebuild Container"
2. Or close and reopen VS Code

### Access terminal

- View → Terminal
- Or press `` Ctrl+` ``

### Install additional packages

```bash
pip install package-name
```

To make permanent, add to `requirements.txt` and rebuild.

### File access

Files in the project directory are shared between host and container. Edit with VS Code or any editor.

## Troubleshooting

**Container fails to build:**
- Check Docker is running
- For GPU config: verify GPU drivers are installed
- Try: "Dev Containers: Rebuild Container Without Cache"
- For Mac users: use CPU configuration

**GPU not detected (GPU configuration):**
- Check `nvidia-smi` works on host
- Verify Docker GPU support is enabled
- Restart Docker Desktop
- Or switch to CPU configuration

**Out of disk space:**
- Clean Docker images: `docker system prune`
- Need at least 15GB free (GPU) or 10GB (CPU)

**Port already in use:**
- Change port in `.devcontainer/gpu/devcontainer.json` or `.devcontainer/cpu/devcontainer.json`

**Wrong configuration selected:**
- Close VS Code
- Reopen folder and choose correct configuration
- Or manually select: `F1` → "Dev Containers: Reopen in Container" → choose configuration
- Rebuild container

## Next steps

- [Student guide](../guides/students.md) - How to participate as a worker
- [Configuration](../guides/configuration.md) - Customize your setup
- [Local training](local-training.md) - Train without database
