# Dev Container Configurations

This project offers two devcontainer configurations:

## GPU Configuration

**Location**: `.devcontainer/gpu/`

**Use when:**
- You have an NVIDIA GPU
- Running on Linux or WSL2
- Docker has GPU support enabled

**Features:**
- Pre-installed PyTorch with CUDA support
- GPU-accelerated training
- Based on `gperdrizet/deeplearning-gpu` image

**Requirements:**
- NVIDIA drivers ≥545
- Docker with GPU support
- 15GB free disk space

## CPU Configuration

**Location**: `.devcontainer/cpu/`

**Use when:**
- Running on Mac (Intel or Apple Silicon)
- No GPU available
- Don't have Docker GPU support
- Just want to develop/test code

**Features:**
- PyTorch CPU-only (auto-installed)
- Works on any machine
- Based on Microsoft's Python 3.10 devcontainer

**Requirements:**
- Docker Desktop
- 10GB free disk space

## How to Choose

When you open this project in VS Code, it will prompt you to select a configuration:

1. **DeepLearning GPU** - for GPU machines
2. **DeepLearning CPU** - for Macs and CPU-only machines

The worker code automatically detects your hardware, so both configurations work seamlessly with the distributed training system.

## Switching Configurations

To switch between configurations:

1. Close VS Code
2. Reopen the project folder
3. Select the other configuration when prompted

Or use `F1` → "Dev Containers: Reopen in Container" → choose configuration
