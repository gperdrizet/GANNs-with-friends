# Distributed GAN Training System Design

## Overview
A distributed GAN training system using students as a compute cluster, with a PostgreSQL database as the communication layer for coordination.

## Architecture

### System components

#### 1. **Main process (coordinator)**
- Runs on instructor's machine
- Manages training loop orchestration
- Aggregates gradients from workers
- Applies optimizer steps
- Tracks training progress
- Updates model weights in database

#### 2. **Worker processes** 
- Runs on student machines (forked repos in devcontainers)
- Fetches work units and current model weights from database
- Performs forward and backward passes on assigned batches
- Uploads gradients back to database
- Polls continuously for new work

#### 3. **PostgreSQL database**
- Central communication hub
- Stores model weights, gradients, work units, and metadata
- No direct networking between main/workers needed

### Training flow

```
Main Process:
1. Initialize training state and model weights in DB
2. Create work units (batches of image indices)
3. Poll for completed gradients from workers
4. When enough gradients collected:
   - Average gradients
   - Apply optimizer step (Adam)
   - Update model weights in DB
   - Mark work units as complete
5. Generate sample images periodically
6. Repeat until convergence

Worker Process:
1. Register in DB with unique worker ID
2. Poll for available work unit
3. Claim work unit (atomic operation)
4. Download current model weights
5. Load assigned images from local dataset
6. For each batch in work unit:
   - Train discriminator on real/fake images
   - Train generator
7. Upload gradients to DB
8. Mark work unit complete
9. Update heartbeat
10. Repeat from step 2
```

### Database schema

#### Tables

**1. model_weights**
- `id`: Primary key
- `model_type`: 'generator' or 'discriminator'
- `iteration`: Training iteration
- `weights_blob`: Serialized state_dict (BYTEA)
- `created_at`: Timestamp

**2. gradients**
- `id`: Primary key
- `worker_id`: Foreign key to workers
- `model_type`: 'generator' or 'discriminator'
- `iteration`: Training iteration
- `gradients_blob`: Serialized gradient tensors (BYTEA)
- `work_unit_id`: Foreign key to work_units
- `created_at`: Timestamp

**3. work_units**
- `id`: Primary key
- `iteration`: Training iteration
- `image_indices`: JSON array of image indices
- `status`: 'pending', 'claimed', 'completed', 'failed'
- `worker_id`: Foreign key (nullable)
- `claimed_at`: Timestamp
- `completed_at`: Timestamp
- `timeout_at`: Timestamp

**4. training_state**
- `id`: Primary key (singleton row)
- `current_iteration`: Current training iteration
- `current_epoch`: Current epoch
- `g_loss`: Generator loss
- `d_loss`: Discriminator loss
- `d_real_acc`: Discriminator accuracy on real images
- `d_fake_acc`: Discriminator accuracy on fake images
- `total_batches_processed`: Counter
- `training_active`: Boolean flag
- `updated_at`: Timestamp

**5. workers**
- `id`: Primary key
- `worker_id`: Unique worker identifier (UUID)
- `hostname`: Worker hostname
- `gpu_name`: GPU device name
- `status`: 'active', 'idle', 'offline'
- `total_work_units`: Counter
- `total_batches`: Counter
- `last_heartbeat`: Timestamp
- `created_at`: Timestamp

### Model architecture

**DCGAN (Deep Convolutional GAN)**
- Target image size: 64x64x3 (CelebA cropped and resized)
- Latent dimension: 100

**Generator:**
```
Input: 100-dim noise vector
- ConvTranspose2d: 100 -> 512 x 4 x 4
- ConvTranspose2d: 512 -> 256 x 8 x 8
- ConvTranspose2d: 256 -> 128 x 16 x 16
- ConvTranspose2d: 128 -> 64 x 32 x 32
- ConvTranspose2d: 64 -> 3 x 64 x 64
Output: 64x64x3 image
Activations: BatchNorm + ReLU (except output: Tanh)
```

**Discriminator:**
```
Input: 64x64x3 image
- Conv2d: 3 -> 64 x 32 x 32
- Conv2d: 64 -> 128 x 16 x 16
- Conv2d: 128 -> 256 x 8 x 8
- Conv2d: 256 -> 512 x 4 x 4
- Conv2d: 512 -> 1 x 1 x 1
Output: Single logit (real/fake)
Activations: BatchNorm + LeakyReLU (except output: none)
```

### Training parameters

- **Batch size per worker**: 32
- **Batches per work unit**: 10 (320 images per work unit)
- **Learning rate**: 0.0002
- **Optimizer**: Adam (β1=0.5, β2=0.999)
- **Loss function**: Binary Cross Entropy
- **Work unit timeout**: 5 minutes
- **Worker heartbeat interval**: 30 seconds
- **Gradient aggregation**: Average across all workers

### Data management

**CelebA Dataset:**
- ~200,000 celebrity face images
- Preprocessed to 64x64 RGB
- Stored locally on each student machine (no network transfer)
- Work units contain only image indices

**Data Download Script:**
- Automated download and preprocessing
- Creates index file for reproducibility
- All workers must have identical dataset order

### Configuration

**config.yaml (in root directory):**
```yaml
database:
  host: <PROVIDED_BY_INSTRUCTOR>
  port: 5432
  database: distributed_gan
  user: <PROVIDED_BY_INSTRUCTOR>
  password: <PROVIDED_BY_INSTRUCTOR>

training:
  batch_size: 32
  batches_per_work_unit: 10
  latent_dim: 100
  image_size: 64
  num_workers_per_update: 3  # Minimum workers before updating
  
worker:
  poll_interval: 5  # seconds
  heartbeat_interval: 30  # seconds
  work_unit_timeout: 300  # seconds
  
data:
  dataset_path: ./data/celeba
  num_workers_dataloader: 4
```

### Implementation files

```
src/
├── models/
│   └── dcgan.py              # Generator and Discriminator models
├── data/
│   └── dataset.py            # CelebA dataset loader
├── database/
│   ├── schema.py             # SQLAlchemy table definitions
│   ├── init_db.py            # Database initialization
│   └── db_manager.py         # Database operations
├── worker.py                 # Worker process
├── main.py                   # Main coordinator
└── utils.py                  # Helper functions

scripts/
└── download_celeba.py        # Dataset download script

config.yaml.template          # Configuration template (in root)
```

### Fault tolerance

**Work Unit Timeout:**
- If worker doesn't complete work unit within timeout, mark as 'failed'
- Main process reassigns failed work units
- Prevents stalled training if worker crashes

**Worker Dropout:**
- Workers update heartbeat regularly
- Main process monitors heartbeats
- Stale workers marked as 'offline'
- Training continues with remaining workers

**New Worker Join:**
- Workers can join anytime
- Claim available/failed work units
- Immediately participate in training

### Monitoring dashboard (future enhancement)

Students can view:
- Their contribution metrics (batches processed, uptime)
- Current training progress (iteration, losses)
- Active workers count
- Sample generated images
- Real-time training curves

## Benefits for educational use

1. **Distributed Systems Concepts**: Students learn about coordination, fault tolerance, race conditions
2. **Database as Message Queue**: Novel use of SQL for distributed computing
3. **Parallel Training**: Understanding data parallelism in deep learning
4. **Real Hardware**: Students see their GPUs contributing to real model training
5. **Collaborative**: Entire class works together towards common goal
6. **Scalable**: Easy to add/remove workers without complex networking

## Performance estimates

**With 10 workers (avg GPU: RTX 3060):**
- ~15-20 batches/sec total throughput
- ~200k images dataset / 32 batch size = 6,250 iterations/epoch
- ~5-6 minutes per epoch
- ~50 epochs for decent results
- **Total training time: ~4-5 hours**

## Future enhancements

1. Web-based monitoring dashboard
2. Gradient compression (reduce DB bandwidth)
3. Support for other GAN architectures (StyleGAN, etc.)
4. Support for other datasets (configurable)
5. Checkpoint saving/resuming
6. TensorBoard integration
7. Automatic hyperparameter tuning
