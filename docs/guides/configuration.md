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
- Auto-reduced for CPU workers

**`batches_per_work_unit`** (integer, default: 10)
- How many batches in each work unit
- Each work unit = batch_size × batches_per_work_unit images
- Larger = less database overhead, more work per claim

**`num_workers_per_update`** (integer, default: 3)
- Wait for N workers before aggregating gradients
- Higher = better gradient quality, slower updates
- Should be ≤ number of active workers

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
  dataset_path: ./data/celeba
  image_size: 64
  normalize: true
```

### Options

**`dataset_path`** (string, default: ./data/celeba)
- Path to CelebA dataset
- Can be relative or absolute
- Must contain img_align_celeba folder

**`image_size`** (integer, default: 64)
- Image resolution (square)
- Must be power of 2
- Typical: 32, 64, 128

**`normalize`** (boolean, default: true)
- Normalize images to [-1, 1]
- Required for DCGAN
- Keep as true

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
  batch_size: 8  # Auto-adjusted by worker
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
