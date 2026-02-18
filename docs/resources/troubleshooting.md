# Troubleshooting

Common issues and solutions for distributed GAN training.

## Installation issues

### Python version too old

**Problem**: `SyntaxError` or `ModuleNotFoundError`

**Solution**:
```bash
# Check version
python --version

# Need Python 3.10+
# Install newer version or use pyenv
```

### PyTorch CUDA mismatch

**Problem**: `RuntimeError: CUDA not available` despite having GPU

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only:
pip install -r .devcontainer/cpu/requirements.txt
```

### Database connection fails

**Problem**: `psycopg2.OperationalError: could not connect`

**Solution**:
```bash
# Verify credentials in config.yaml
# Test connection manually
psql -h HOST -U USER -d DATABASE

# Check firewall allows port 5432
# Verify database is publicly accessible
```

## Runtime issues

### Out of memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 16  # or 8

# Close other GPU applications
# Check GPU memory: nvidia-smi
# Try CPU-only mode
```

### Worker can't find dataset

**Problem**: `FileNotFoundError: data/celeba not found`

**Solution**:
```bash
# Download dataset
python scripts/download_celeba.py

# Verify location
ls data/celeba/img_align_celeba/

# Check config.yaml path
data:
  dataset_path: ./data/celeba
```

### No work units available

**Problem**: Worker polls but finds no work

**Solution**:
- Wait for coordinator to start
- Check coordinator is creating work units
- Verify database connection
- Check current iteration matches

```sql
-- Check work units exist
SELECT COUNT(*), status FROM work_units GROUP BY status;
```

### Work units timeout

**Problem**: Work units marked as stalled and reclaimed

**Solution**:
```yaml
# Increase timeout in config.yaml
worker:
  work_unit_timeout: 600  # 10 minutes

# Check worker performance
# May need to reduce batch size
# Check network speed
```

## Training issues

### Loss values are NaN

**Problem**: Generator or discriminator loss shows NaN

**Solution**:
```yaml
# Reduce learning rates
training:
  learning_rate_g: 0.0001
  learning_rate_d: 0.0001

# Check for bad gradients
# Restart training from checkpoint
# Verify dataset loaded correctly
```

### Poor image quality

**Problem**: Generated images look like noise

**Solution**:
- Train longer (more epochs)
- Check loss values are decreasing
- Verify dataset images look correct
- Try different hyperparameters

```bash
# Check sample images
ls data/outputs/samples/
```

### Training very slow

**Problem**: Iterations take very long

**Solution**:
- Need more active workers
- Check database performance
- Verify network speed
- Increase `num_workers_per_update` to gather more gradients

### Discriminator dominates

**Problem**: Discriminator loss → 0, generator loss increases

**Solution**:
```yaml
# Lower discriminator learning rate
training:
  learning_rate_g: 0.0002
  learning_rate_d: 0.0001

# Or increase generator learning rate
# Common in GAN training
```

## Database issues

### Database full

**Problem**: `ERROR: disk full`

**Solution**:
```sql
-- Check table sizes
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables
WHERE schemaname = 'public';

-- Clean old gradients
DELETE FROM gradients
WHERE uploaded_at < NOW() - INTERVAL '1 day';

-- Vacuum database
VACUUM FULL;
```

### Too many connections

**Problem**: `FATAL: too many connections`

**Solution**:
```sql
-- Check current connections
SELECT COUNT(*) FROM pg_stat_activity;

-- Increase max_connections in postgresql.conf
max_connections = 100

-- Or use connection pooling
```

### Slow queries

**Problem**: Database queries take long time

**Solution**:
```sql
-- Add indexes
CREATE INDEX idx_work_units_status ON work_units(status, iteration);
CREATE INDEX idx_workers_heartbeat ON workers(last_heartbeat);

-- Analyze tables
ANALYZE work_units;
ANALYZE workers;
```

## Network issues

### Timeouts

**Problem**: Frequent connection timeouts

**Solution**:
```yaml
# Increase poll interval
worker:
  poll_interval: 10  # seconds

# Check network stability
# ping DATABASE_HOST

# Use closer database region
```

### Slow uploads

**Problem**: Gradient upload takes very long

**Solution**:
- Check network speed
- Database may be far away geographically
- Consider compressing gradients
- Use closer database provider

## Colab-specific issues

### Session disconnects

**Problem**: Colab session times out

**Solution**:
- Keep browser tab active
- Don't idle too long
- Use Colab Pro for longer sessions
- Re-run cells to resume

### GPU quota exceeded

**Problem**: Can't get GPU runtime

**Solution**:
- Wait a few hours for quota reset
- Use CPU runtime temporarily
- Consider Colab Pro
- Try at different time of day

### Files disappear

**Problem**: Dataset or config lost after disconnect

**Solution**:
```python
# Save important files to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy config
!cp config.yaml /content/drive/MyDrive/
```

## Development issues

### Import errors

**Problem**: `ModuleNotFoundError` for project modules

**Solution**:
```python
# Ensure in correct directory
import os
os.chdir('/path/to/GANNs-with-freinds')

# Or add to path
import sys
sys.path.insert(0, '/path/to/GANNs-with-freinds/src')
```

### Git issues

**Problem**: Can't push/pull changes

**Solution**:
```bash
# Stash local changes
git stash

# Pull updates
git pull

# Reapply changes
git stash pop

# Or create branch
git checkout -b my-changes
```

## Debugging techniques

### Enable debug logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check GPU utilization

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Should show:
# - GPU utilization > 0%
# - Memory usage increasing during training
```

### Verify data loading

```python
from src.data.dataset import CelebADataset

dataset = CelebADataset('data/celeba', image_size=64)
print(f'Dataset size: {len(dataset)}')

# Load sample
image, _ = dataset[0]
print(f'Image shape: {image.shape}')  # Should be [3, 64, 64]
```

### Test database connection

```python
from src.utils import load_config, build_db_url
import psycopg2

config = load_config('config.yaml')
db_url = build_db_url(config['database'])

try:
    conn = psycopg2.connect(db_url)
    print('Database connection successful!')
    conn.close()
except Exception as e:
    print(f'Connection failed: {e}')
```

### Check model initialization

```python
from src.models.dcgan import Generator, Discriminator
import torch

gen = Generator(latent_dim=100)
disc = Discriminator()

# Test forward pass
noise = torch.randn(1, 100, 1, 1)
fake_images = gen(noise)
print(f'Generated image shape: {fake_images.shape}')'  # [1, 3, 64, 64]

output = disc(fake_images)
print(f'Discriminator output shape: {output.shape}')  # [1, 1]
```

## Getting help

### Check logs

```bash
# Worker logs
python src/worker.py 2>&1 | tee worker.log

# Coordinator logs
python src/main.py 2>&1 | tee coordinator.log
```

### Create minimal example

Isolate the issue:
```python
# Minimal reproduction
import torch
from src.models.dcgan import Generator

gen = Generator()
noise = torch.randn(1, 100, 1, 1)
output = gen(noise)
```

### Report issue

Include:
- Error message
- Steps to reproduce
- System information (OS, Python version, GPU)
- Relevant config settings

## Preventive measures

- Start with small test run (few epochs)
- Monitor initially before leaving overnight
- Keep checkpoints frequently
- Back up database regularly
- Test with one worker before scaling
- Verify dataset integrity
- Use version control

## Next steps

- [Performance optimization](performance.md)
- [FAQ](faq.md)
- [Configuration reference](../guides/configuration.md)
