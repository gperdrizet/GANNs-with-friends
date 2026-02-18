# Local training setup

Train the GAN on a single GPU without the distributed setup.

## When to use local training

Perfect for:
- Experimentation and development
- Testing hyperparameters
- Baseline performance comparison
- Learning GANs independently
- No database access needed

## Trade-offs

**Advantages:**
- Simpler setup (no database)
- Faster iteration
- Complete control
- Immediate feedback

**Disadvantages:**
- Single GPU only
- Miss distributed systems lessons
- Can't collaborate with class

## Prerequisites

- Any installation path completed (Colab, dev container, native, or conda)
- CelebA dataset downloaded
- **No database required**

## Quick start

### 1. Skip database configuration

You don't need `config.yaml` for local training.

### 2. Start training

```bash
python src/train_local.py --epochs 50 --batch-size 128
```

Training begins immediately:
```
Starting local DCGAN training...
Device: cuda
Dataset: 202,599 images
Generator parameters: 4,683,971
Discriminator parameters: 2,765,249

Epoch 1/50
[1/1,582] Loss_D: 1.234 Loss_G: 2.345
[2/1,582] Loss_D: 1.123 Loss_G: 2.234
...
```

## Command-line options

### Basic options

```bash
python src/train_local.py \
  --epochs 50 \                    # Number of epochs
  --batch-size 128 \               # Batch size
  --dataset-path data/celeba \     # Dataset location
  --output-dir outputs_local       # Output directory
```

### Advanced options

```bash
python src/train_local.py \
  --epochs 100 \
  --batch-size 64 \
  --lr-g 0.0002 \                  # Generator learning rate
  --lr-d 0.0002 \                  # Discriminator learning rate
  --latent-dim 100 \               # Latent space dimension
  --image-size 64 \                # Image resolution
  --sample-interval 1 \            # Generate samples every N epochs
  --checkpoint-interval 5 \        # Save checkpoint every N epochs
  --num-workers 4                  # DataLoader workers
```

### Resume training

Continue from a checkpoint:

```bash
python src/train_local.py \
  --resume outputs_local/checkpoints/checkpoint_epoch_0025.pth \
  --epochs 100
```

## Monitoring progress

### Generated samples

View generated images during training:

```bash
ls outputs_local/samples/
# epoch_001.png
# epoch_002.png
# ...
```

Open these images to see the generator improving.

### Checkpoints

Model checkpoints saved periodically:

```bash
ls outputs_local/checkpoints/
# checkpoint_epoch_0005.pth
# checkpoint_epoch_0010.pth
# checkpoint_latest.pth
```

### Console output

Training prints loss values:

```
Epoch 5/50
Avg Generator Loss: 2.145
Avg Discriminator Loss: 0.823
Time: 45.2s
```

## Viewing results

### Use the demo notebook

After training (or during):

```bash
jupyter notebook notebooks/demo_trained_model.ipynb
```

The notebook can load local checkpoints:

```python
# In notebook, point to local checkpoint
checkpoint_path = '../outputs_local/checkpoints/checkpoint_latest.pth'
```

### Manual inspection

```python
import torch
from src.models.dcgan import Generator

# Load checkpoint
checkpoint = torch.load('outputs_local/checkpoints/checkpoint_latest.pth')

# Create generator
generator = Generator(latent_dim=100)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# Generate images
noise = torch.randn(16, 100, 1, 1)
with torch.no_grad():
    images = generator(noise)
```

## Comparing with distributed training

### Performance comparison

**Local training (single GPU):**
- Full batch every step
- Immediate gradient updates
- Faster per-iteration

**Distributed training:**
- Combined gradients from N workers
- More diverse batches per update
- Better generalization (often)

### Try both

1. Train locally first to understand GANs
2. Then participate in distributed training
3. Compare results and learning experience

## Hyperparameter tuning

Experiment with different settings:

### Learning rates

```bash
# Default
python src/train_local.py --lr-g 0.0002 --lr-d 0.0002

# Lower (more stable)
python src/train_local.py --lr-g 0.0001 --lr-d 0.0001

# Higher (faster but risky)
python src/train_local.py --lr-g 0.0004 --lr-d 0.0004
```

### Batch sizes

```bash
# Smaller (more updates)
python src/train_local.py --batch-size 64

# Default
python src/train_local.py --batch-size 128

# Larger (more stable)
python src/train_local.py --batch-size 256
```

### Training duration

```bash
# Quick test (5 epochs)
python src/train_local.py --epochs 5

# Standard
python src/train_local.py --epochs 50

# Extended
python src/train_local.py --epochs 200
```

## Troubleshooting

**Out of memory:**
- Reduce `--batch-size`
- Reduce `--num-workers`
- Use `--image-size 32` for smaller images

**Training unstable:**
- Lower learning rates
- Try different batch sizes
- Check loss values (both should decrease)

**Poor quality images:**
- Train longer (more epochs)
- Adjust learning rates
- Verify dataset loaded correctly

**Slow training:**
- Increase `--num-workers`
- Use GPU (verify with `nvidia-smi`)
- Increase batch size if memory allows

## Next steps

- [Architecture overview](../architecture/models.md) - Understand the models
- [Distributed training](../features/distributed-training.md) - Try collaborative training
- [Performance tips](../resources/performance.md) - Optimize training
