# FAQ

Frequently asked questions about distributed GAN training.

## General questions

### What is this project about?

An educational distributed deep learning system where students collectively train a GAN to generate celebrity faces. It teaches distributed systems, GANs, and collaborative computing.

### Do I need a GPU?

No. The system works with CPU-only workers (automatically adjusted batch sizes). GPU is faster but optional.

### How long does training take?

Depends on:
- Number of active workers
- Hardware (GPU vs CPU)
- Target quality

Typical: 2-6 hours with 5-10 GPU workers for decent results.

### Can I join training late?

Yes! Workers can join anytime. Just start your worker and it will begin contributing.

### What if I disconnect?

No problem. Your work is saved. Restart your worker and continue. Training state is in the database.

## Setup questions

### Which installation path should I choose?

- **Google Colab**: Easiest, no local setup, free GPU
- **Dev container**: Best for development, requires Docker
- **Native Python**: Direct control, no Docker needed
- **Conda**: If you already use conda

See [installation guide](../getting-started/installation.md) for details.

### Do I need to download the dataset?

Yes. The CelebA dataset (~1.4 GB) is required. Run:
```bash
python scripts/download_celeba.py
```

### Where do I get database credentials?

Your instructor provides these. You need:
- Host address
- Database name
- Username
- Password

### Can I use my own database?

Yes, but you'd be the coordinator, not a worker. See [instructor guide](../guides/instructors.md).

## Training questions

### How do I know it's working?

Your worker prints progress:
```
Processing work unit 42...
Completed work unit 42 in 15.3s
```

Check with instructor or view results on Hugging Face.

### What do the loss values mean?

- **Generator loss**: How badly the generator fools the discriminator (lower = better fooling)
- **Discriminator loss**: How well discriminator distinguishes real vs fake (lower = better discrimination)

Both should generally trend downward but fluctuate.

### Why are my loss values different from others?

Different work units have different batches. Loss varies across batches. Look at trends, not individual values.

### When will I see results?

- Samples generated every N iterations (instructor sets this)
- Results improve over time
- Check `outputs/samples/` or Hugging Face
- Realistic faces emerge after many iterations

### Can I train my own model?

Yes! Use local training mode:
```bash
python src/train_local.py --epochs 50
```

No database needed. See [local training guide](../setup/local-training.md).

## Technical questions

### How does coordination work?

Workers poll the PostgreSQL database for work units, compute gradients, upload results. Coordinator aggregates gradients and updates weights. See [architecture overview](../architecture/overview.md).

### What happens if my worker crashes?

The work unit times out and is automatically reclaimed. Another worker (or you after restarting) will process it. No data loss.

### Can I run multiple workers?

Yes, if you have multiple GPUs. Create separate config files or run in different directories.

### How is this different from PyTorch DDP?

PyTorch DDP requires direct network communication. This uses database coordination, making it easier for distributed educational setups across networks/firewalls.

### What's the database storing?

- Model weights (current and historical)
- Computed gradients from workers
- Work unit assignments and status
- Worker registration and statistics

See [database schema](../architecture/database.md).

## Performance questions

### Why is training slow?

- Need more workers
- Workers may have slow hardware
- Network latency to database
- Check batch sizes and configuration

### How can I speed it up?

- Recruit more workers
- Use GPU not CPU
- Optimize database location (closer geographically)
- Increase batch sizes (if memory allows)

### What's the optimal number of workers?

No hard limit. More workers = faster training (with diminishing returns). Typical: 5-15 workers for class project.

### Does CPU training help?

Yes! CPU workers contribute, though slower than GPU. Every worker helps.

## Results questions

### How do I view generated faces?

Options:
1. Check `outputs/samples/` (if coordinator)
2. Open demo notebook: `notebooks/demo_trained_model.ipynb`
3. View on Hugging Face (if enabled)

### Why do images look blurry?

Early in training, images are noisy/blurry. Quality improves with more iterations. Check latest samples, not early ones.

### Can I generate my own faces?

Yes! After training:
```python
from src.models.dcgan import Generator
import torch

gen = Generator()
# Load trained weights
checkpoint = torch.load('checkpoint.pth')
gen.load_state_dict(checkpoint['generator_state_dict'])

# Generate
noise = torch.randn(16, 100, 1, 1)
faces = gen(noise)
```

### How do I save my favorite generated faces?

```python
from PIL import Image
import torchvision.transforms as T

# Convert tensor to image
to_pil = T.ToPILImage()
image = to_pil(face_tensor)
image.save('my_face.png')
```

## Troubleshooting questions

### Worker says "no work units available"

- Training may not have started
- Current iteration completed, next not yet created
- Check with instructor

See [troubleshooting guide](troubleshooting.md).

### Getting connection errors

- Verify credentials in `config.yaml`
- Check network connection
- Database may be down (ask instructor)

### Out of memory errors

- Reduce batch size in `config.yaml`
- Close other GPU applications
- Try CPU-only mode

### Loss is NaN

- Lower learning rates
- Restart from last checkpoint
- May indicate training instability

## Educational questions

### What will I learn?

- Distributed system architecture
- Database-coordinated computing
- GAN training and theory
- Collaborative problem solving
- Python and PyTorch

### Do I need to understand all the code?

No. Workers can participate with basic understanding. Deeper learning comes from exploration.

### Can I modify the code?

Yes! This is open source and educational. Try:
- Different model architectures
- Alternative optimizers
- Custom loss functions
- Enhanced monitoring

### Where can I learn more about GANs?

- [DCGAN paper](https://arxiv.org/abs/1511.06434)
- [GAN training tips](https://github.com/soumith/ganhacks)
- PyTorch GAN tutorials
- This project's architecture docs

## Contributing questions

### Can I contribute improvements?

Yes! This is an educational project. Contributions welcome:
- Bug fixes
- Documentation improvements
- New features
- Performance optimizations

See [contributing guide](contributing.md).

### I found a bug, what do I do?

- Create GitHub issue with details
- Include error messages and steps to reproduce
- Or submit a pull request with fix

### Can I use this for my research?

Yes, with attribution. See LICENSE file. Consider citing if you publish results.

## Advanced questions

### How do I add Hugging Face integration?

Instructor sets this up. See [Hugging Face integration guide](../features/huggingface-integration.md).

### Can I use different datasets?

Yes, but requires code modifications:
- Implement custom dataset class
- Update data loader
- Adjust image size if needed

### How do I modify the GAN architecture?

Edit `src/models/dcgan.py`:
- Change layer sizes
- Add/remove layers
- Try different architectures (StyleGAN, etc.)

### Can this scale to 100+ workers?

Yes, but may need:
- Database optimization
- Connection pooling
- Higher-capacity database server
- Gradient compression

## Still have questions?

- Check [troubleshooting guide](troubleshooting.md)
- Ask your instructor
- Create GitHub issue
- Explore the [architecture documentation](../architecture/overview.md)
