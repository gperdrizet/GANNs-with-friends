# Configuration reference

Complete reference for all configuration options in `config.yaml`.

## Configuration file structure

```yaml
database:
  # Database connection settings
  
training:
  # Training hyperparameters
  
worker:
  # Worker behavior settings
  
model:
  # Model architecture settings
  
data:
  # Dataset configuration
  
huggingface:
  # Hugging Face integration (optional)
```

## Database configuration

Connection settings for PostgreSQL database.

```yaml
database:
  host: localhost
  port: 5432
  database: distributed_gan
  user: username
  password: password
```

### Options

**`host`** (string, required)
- Database server address
- Can be IP address or hostname
- Examples: `localhost`, `db.example.com`, `192.168.1.100`

**`port`** (integer, default: 5432)
- PostgreSQL port number
- Standard PostgreSQL port is 5432

**`database`** (string, required)
- Database name
- Must exist before running

**`user`** (string, required)
- Database username
- Needs SELECT, INSERT, UPDATE permissions

**`password`** (string, required)
- Database password
- Keep secure, don't commit to version control

## Training configuration

Hyperparameters for training process.

```yaml
training:
  batch_size: 32
  batches_per_work_unit: 10
  num_workers_per_update: 3
  learning_rate_g: 0.0002
  learning_rate_d: 0.0002
  beta1: 0.5
  beta2: 0.999
  num_epochs: 50
```

### Options

**`batch_size`** (integer, default: 32)
- Number of images per batch
- Larger = more stable but more memory
- Typical range: 16-128
- Reduce if you encounter out-of-memory errors

**`batches_per_work_unit`** (integer, default: 10)
- How many batches in each work unit
- Each work unit = batch_size × batches_per_work_unit images
- Larger = less database overhead, more work per claim

**`num_workunits_per_update`** (integer, default: 3)
- Wait for N work unit gradients before aggregating and updating models
- Higher = more gradient samples, better convergence, slower updates
- Lower = faster iteration, but potentially noisier gradients
- Should be set based on your total number of workers and dataset size

**`learning_rate_g`** (float, default: 0.0002)
- Generator learning rate
- Lower = more stable, slower learning
- Typical range: 0.0001-0.0004

**`learning_rate_d`** (float, default: 0.0002)
- Discriminator learning rate
- Often same as generator
- Can tune independently

**`beta1`** (float, default: 0.5)
- Adam optimizer beta1 parameter
- Controls momentum
- 0.5 is standard for GAN training

**`beta2`** (float, default: 0.999)
- Adam optimizer beta2 parameter
- Controls variance
- Usually keep at 0.999

**`num_epochs`** (integer, default: 50)
- Total epochs to train
- One epoch = all images seen once
- More epochs = longer training

## Worker configuration

Settings for worker behavior.

```yaml
worker:
  poll_interval: 5
  heartbeat_interval: 30
  work_unit_timeout: 300
  max_retries: 3
  num_workers_dataloader: 4
```

### Options

**`poll_interval`** (integer, default: 5)
- Seconds between database polls
- Lower = more responsive, more database load
- Typical range: 1-10

**`heartbeat_interval`** (integer, default: 30)
- Seconds between heartbeat updates
- Shows worker is still alive
- Typical range: 15-60

**`work_unit_timeout`** (integer, default: 300)
- Seconds before uncompleted work is reclaimed
- Should exceed normal processing time
- Typical range: 120-600

**`max_retries`** (integer, default: 3)
- Max attempts to process a work unit
- Prevents infinite retry on bad data
- Typical range: 3-5

**`num_workers_dataloader`** (integer, default: 4)
- PyTorch DataLoader worker processes
- Higher = faster data loading, more memory
- Set to 0 for debugging

## Model configuration

Model architecture parameters.

```yaml
model:
  latent_dim: 100
  generator_features: 64
  discriminator_features: 64
```

### Options

**`latent_dim`** (integer, default: 100)
- Dimension of random noise vector
- Standard DCGAN uses 100
- Higher = more capacity, slower

**`generator_features`** (integer, default: 64)
- Base number of generator feature maps
- Determines model size
- Higher = more capacity, more memory

**`discriminator_features`** (integer, default: 64)
- Base number of discriminator feature maps
- Usually same as generator
- Higher = better discrimination, more memory

## Data configuration

Dataset settings.

```yaml
data:
  dataset_path: data/celeba_torchvision/celeba/img_align_celeba
  num_workers_dataloader: 4
```

### Options

**`dataset_path`** (string, default: data/celeba_torchvision/celeba/img_align_celeba)
- Path to CelebA dataset images
- If the dataset is not found at this path, it will be automatically downloaded from Hugging Face
- Can be relative or absolute

**`num_workers_dataloader`** (integer, default: 4)
- Number of dataloader workers for parallel data loading
- Set to 0 to disable multiprocessing

## Hugging Face configuration

Optional integration for model sharing.

```yaml
huggingface:
  enabled: false
  repo_id: ''
  token: ''
  push_interval: 5
  private: false
```

### Options

**`enabled`** (boolean, default: false)
- Enable Hugging Face uploads
- Requires valid token and repo

**`repo_id`** (string, default: '')
- Hugging Face repository ID
- Format: username/repo-name
- Example: 'instructor/distributed-gan'

**`token`** (string, default: '')
- Hugging Face access token
- Get from huggingface.co/settings/tokens
- Needs write permissions

**`push_interval`** (integer, default: 5)
- Push checkpoint every N iterations
- Lower = more frequent updates, more uploads
- Typical range: 1-10

**`private`** (boolean, default: false)
- Make repository private
- Students need access to view
- Public repos don't consume quota

## Understanding distributed training tradeoffs

The distributed training system coordinates multiple workers through work units and gradient aggregation. Understanding the tradeoffs helps you configure the system effectively.

### Work unit size vs. database overhead

**`batches_per_work_unit`** controls how many batches of training data each work unit contains:

```yaml
training:
  batch_size: 32
  batches_per_work_unit: 10  # Each work unit = 32 × 10 = 320 images
```

**Larger work units (15-20 batches):**
- Less database overhead (fewer queries per epoch)
- Fewer work units to manage
- Better for stable, persistent workers
- Longer processing time per work unit
- Slower feedback if workers disconnect

**Smaller work units (5-10 batches):**
- Faster completion times
- Better for unstable workers (less wasted work if disconnected)
- More granular progress tracking
- More database operations
- Higher coordination overhead with many workers

**Recommendation:** Start with 10, increase if you have stable workers or high database latency.

### Aggregation threshold vs. gradient quality

**`num_workunits_per_update`** controls how many work unit gradients are collected before updating the model:

```yaml
training:
  num_workunits_per_update: 5  # Wait for 5 work unit gradients
```

This is one of the most important parameters for distributed training quality.

**Higher values (8-20+ work units):**
- More gradient samples = better quality, less noisy updates
- More robust training (similar to larger batch sizes)
- Less risk of mode collapse
- Slower iterations (wait for more workers to finish)
- More wasted work units if not all are used
- Can accumulate stale work units

**Lower values (1-3 work units):**
- Faster iterations (update as soon as possible)
- Less wasted computation
- Quick feedback during development/testing
- Noisier gradients (more variance in updates)
- Higher risk of training instability
- May not benefit from parallel workers

**The stale work unit problem:**

When `num_workunits_per_update` is less than the total workers, some work units will be "left behind" when the coordinator aggregates and moves to the next iteration:

```
Iteration 1: Create 100 work units
- Wait for 5 to complete (num_workunits_per_update=5)
- Aggregate and move to iteration 2
- 95 work units are now "stale" (cancelled automatically)
```

The system automatically cancels pending work units when advancing iterations to prevent workers from processing stale data. Workers who claim cancelled work units will skip them and move to the next one.

**Guidelines for setting num_workunits_per_update:**

| Class Size | Workers Expected | Recommended Value | Rationale |
|------------|------------------|-------------------|-----------|
| Small (2-5) | 2-5 | 2-3 | Get updates quickly, most workers contribute |
| Medium (10-20) | 10-15 | 5-8 | Balance quality and speed |
| Large (30+) | 20-30 | 10-20 | Higher quality gradients, can afford to wait |

**Testing/Development:** Set to 1 for fastest feedback, but expect noisy training.

**Production training:** Set based on expected worker count and desired gradient quality. A good rule: 30-50% of your typical concurrent workers.

### Update frequency vs. convergence speed

The actual update frequency depends on both parameters:

```
Images per update = batch_size × batches_per_work_unit × num_workunits_per_update

Example with defaults:
32 × 10 × 5 = 1,600 images per model update
```

**More frequent updates (fewer images):**
- Faster iterations through the epoch
- More opportunities to correct course
- Higher overhead from weight synchronization
- Can be noisier

**Less frequent updates (more images):**
- More stable gradient estimates
- Less synchronization overhead
- Slower to respond to training issues
- Similar to traditional large-batch training

**Finding the balance:**

1. Start with defaults for your class size
2. Monitor training metrics (loss, sample quality)
3. If training is unstable: increase `num_workunits_per_update`
4. If training is too slow: decrease `num_workunits_per_update` or `batches_per_work_unit`
5. Adjust based on worker reliability and network conditions

### Worker coordination patterns

Different parameter combinations create different worker coordination patterns:

**Pattern 1: Fast iteration (testing)**
```yaml
batches_per_work_unit: 5
num_workunits_per_update: 1
```
- Single worker can drive training
- Fast feedback, noisy gradients
- Good for debugging, not production

**Pattern 2: Balanced (small/medium class)**
```yaml
batches_per_work_unit: 10
num_workunits_per_update: 5
```
- 5 workers contribute per update
- Good balance of quality and speed
- Default configuration

**Pattern 3: High quality (large class)**
```yaml
batches_per_work_unit: 15
num_workunits_per_update: 15
```
- Wait for many gradient samples
- Best gradient quality
- Slower but more stable training

**Pattern 4: Efficient (stable workers)**
```yaml
batches_per_work_unit: 20
num_workunits_per_update: 10
```
- Maximize work per database operation
- Assumes workers can handle larger units
- Good for low-latency networks

### Monitoring and adjustment

Watch these metrics to tune your configuration:

1. **Work unit completion rate**: If workers finish faster than coordinator aggregates, you might want higher `num_workunits_per_update`

2. **Cancelled work units**: High cancellation rate means too many work units created or `num_workunits_per_update` too low

3. **Worker idle time**: If workers wait often for new work units, reduce `batches_per_work_unit` or `num_workunits_per_update`

4. **Training stability**: If loss oscillates wildly, increase `num_workunits_per_update` for better gradients

5. **Sample quality**: If samples don't improve, try different aggregation thresholds

## Example configurations

### Small class (2-5 students)

```yaml
training:
  batch_size: 64
  batches_per_work_unit: 5
  num_workers_per_update: 2
  num_epochs: 30
```

### Large class (10+ students)

```yaml
training:
  batch_size: 32
  batches_per_work_unit: 10
  num_workers_per_update: 5
  num_epochs: 50
```

### CPU-only mode

```yaml
training:
  batch_size: 8
  batches_per_work_unit: 5
  num_workers_per_update: 3
```

### Quick testing

```yaml
training:
  batch_size: 16
  batches_per_work_unit: 2
  num_workers_per_update: 1
  num_epochs: 5
```

### High quality (long training)

```yaml
training:
  batch_size: 64
  batches_per_work_unit: 10
  num_workers_per_update: 5
  num_epochs: 200
  learning_rate_g: 0.0001
  learning_rate_d: 0.0001
```

## Environment variables

Some settings can be overridden with environment variables:

```bash
# Database password (more secure than config file)
export DB_PASSWORD=secret

# Hugging Face token
export HF_TOKEN=hf_...

# Override config file
export CONFIG_PATH=/path/to/custom/config.yaml
```

## Security best practices

**Don't commit secrets:**
```bash
# Add to .gitignore
config.yaml
.env
```

**Use environment variables:**
```yaml
database:
  password: ${DB_PASSWORD}

huggingface:
  token: ${HF_TOKEN}
```

**Create template:**
```bash
# Provide template without secrets
cp config.yaml config.yaml.template

# Remove sensitive data from template
sed -i 's/password: .*/password: YOUR_PASSWORD/' config.yaml.template
```

## Validation

Validate your config:

```python
python -c "from src.utils import load_config; load_config('config.yaml'); print('Config OK')"
```

## Next steps

- [Quick start](../getting-started/quick-start.md) - Use your configuration
- [Troubleshooting](../resources/troubleshooting.md) - Fix config issues
-[Performance tuning](../resources/performance.md) - Optimize settings
