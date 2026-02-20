# Architecture overview

Understanding the system architecture of distributed GAN training.

## Distributed deep learning fundamentals

Training deep neural networks is computationally expensive. A single forward and backward pass through a model like ours requires ~2.6 billion floating-point operations per image. Multiply that by millions of images and hundreds of epochs, and you're looking at days or weeks of training on a single GPU.

**Distributed training** solves this by spreading the work across multiple machines. The key insight: gradient computation is embarrassingly parallel - each worker can process different images independently.

### Data parallelism

Our system uses **data parallelism**, the most common distributed training strategy:

1. Each worker has a complete copy of the model
2. Workers process different subsets of the training data
3. Workers compute gradients independently
4. Gradients are averaged across workers
5. All workers update to the same new weights

This approach scales well because:
- Adding workers increases throughput linearly
- Communication overhead is proportional to model size, not data size
- Workers don't need to communicate with each other (only with coordinator)

### Synchronous vs asynchronous training

**Synchronous** (our approach): Wait for N workers before updating weights
- Pro: Deterministic, stable convergence
- Con: Slowest worker limits throughput

**Asynchronous**: Workers update weights independently
- Pro: No waiting, maximum throughput
- Con: Stale gradients, harder to converge

We use **partial synchronization** - waiting for a threshold number of workers (not all). This balances consistency with fault tolerance.

## System components

The system consists of four main components:

1. **Coordinator** - Main training process (instructor runs this)
2. **Workers** - Distributed compute nodes (students run these)
3. **PostgreSQL Database** - Coordination and communication hub
4. **Shared Model** - GAN being trained collaboratively

## High-level architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Coordinator (Instructor)                     │
│  ┌──────────────┐   ┌───────────────┐   ┌───────────────────┐  │
│  │ Create Work  │   │   Aggregate   │   │  Optimizer Step   │  │
│  │ Units (320   │ → │   Gradients   │ → │  & Save Weights   │  │
│  │ images each) │   │ (wait for 3)  │   │  to Database      │  │
│  └──────────────┘   └───────────────┘   └───────────────────┘  │
│          │                  ↑                     │              │
└──────────┼──────────────────┼─────────────────────┼─────────────┘
           ↓                  │                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PostgreSQL Database                         │
│                     (perdrizet.org:54321)                        │
│  ┌───────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │   Work    │  │ Gradients │  │  Model   │  │  Training    │  │
│  │   Units   │  │ (per work │  │ Weights  │  │    State     │  │
│  │ (pending/ │  │   unit)   │  │ (G + D)  │  │ (iteration,  │  │
│  │ claimed)  │  │           │  │          │  │  epoch)      │  │
│  └───────────┘  └───────────┘  └──────────┘  └──────────────┘  │
└──────────┬──────────────┬───────────────────────┬───────────────┘
           │              │                       │
           │              │ (poll & claim)        │ (download weights)
           ↓              ↓                       ↓
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│     Worker 1     │  │     Worker 2     │  │     Worker N     │
│ (Student Laptop) │  │  (Colab GPU)     │  │ (Lab Computer)   │
│                  │  │                  │  │                  │
│ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌──────────────┐ │
│ │ Load images  │ │  │ │ Load images  │ │  │ │ Load images  │ │
│ │ (from work   │ │  │ │ (from work   │ │  │ │ (from work   │ │
│ │  unit)       │ │  │ │  unit)       │ │  │ │  unit)       │ │
│ ├──────────────┤ │  │ ├──────────────┤ │  │ ├──────────────┤ │
│ │ Forward pass │ │  │ │ Forward pass │ │  │ │ Forward pass │ │
│ │ Backward pass│ │  │ │ Backward pass│ │  │ │ Backward pass│ │
│ ├──────────────┤ │  │ ├──────────────┤ │  │ ├──────────────┤ │
│ │ Upload grads │ │  │ │ Upload grads │ │  │ │ Upload grads │ │
│ │ to database  │ │  │ │ to database  │ │  │ │ to database  │ │
│ └──────────────┘ │  │ └──────────────┘ │  │ └──────────────┘ │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

**Key configuration parameters** (from `config.yaml`):
- `images_per_work_unit`: 320 images assigned per work unit
- `num_workunits_per_update`: 3 work units must complete before weight update
- `worker.batch_size`: Each worker processes images in batches (default: 32)

## Data flow

### 1. Initialization (Coordinator)

```
Coordinator starts
├─> Initialize generator and discriminator with random weights
├─> Apply DCGAN weight initialization (normal distribution)
├─> Save initial weights to database (iteration 0)
├─> Create work units for iteration 1
│   └─> Each work unit = list of image indices (320 images)
└─> Set training_state (iteration=1, epoch=1)
```

### 2. Worker claims and processes

```
Worker polls database
├─> Find unclaimed work unit for current iteration
├─> Atomically claim it (FOR UPDATE SKIP LOCKED)
├─> Download current model weights from database
├─> Load assigned images from CelebA dataset
├─> Process images in batches (batch_size from worker config)
│   ├─> For each batch:
│   │   ├─> Train discriminator on real images
│   │   ├─> Train discriminator on fake images
│   │   ├─> Train generator to fool discriminator
│   │   └─> Accumulate gradients
├─> Average accumulated gradients
├─> Upload gradients to database
└─> Mark work unit as completed
```

### 3. Coordinator aggregates

```
Coordinator waits for N work units (num_workunits_per_update)
├─> Check completed work units for current iteration
├─> When N work units done:
│   ├─> Download gradient tensors from all N work units
│   ├─> Weighted average gradients (by num_samples)
│   ├─> Apply gradients to model parameters
│   ├─> Run optimizer step (Adam)
│   ├─> Save updated weights to database
│   ├─> Cancel remaining pending work units (stale)
│   ├─> Increment iteration counter
│   └─> Create work units for next iteration
└─> Repeat until epoch complete
```

### 4. Iteration continues

```
Loop until training complete:
├─> Workers claim new work units
├─> Coordinator aggregates new gradients
├─> Generate sample images periodically
├─> Save checkpoints
└─> Push to Hugging Face (optional)
```

## Database schema

### training_state table

Stores current training status:

```sql
CREATE TABLE training_state (
    id SERIAL PRIMARY KEY,
    current_iteration INTEGER,
    current_epoch INTEGER,
    generator_weights BYTEA,        -- Serialized PyTorch tensor
    discriminator_weights BYTEA,
    optimizer_g_state BYTEA,
    optimizer_d_state BYTEA,
    generator_loss REAL,
    discriminator_loss REAL,
    updated_at TIMESTAMP
);
```

### work_units table

Individual work assignments:

```sql
CREATE TABLE work_units (
    id SERIAL PRIMARY KEY,
    iteration INTEGER,
    epoch INTEGER,
    start_index INTEGER,           -- First image index
    end_index INTEGER,             -- Last image index
    status VARCHAR(20),            -- pending/in_progress/completed/failed
    claimed_by VARCHAR(100),       -- Worker ID
    claimed_at TIMESTAMP,
    completed_at TIMESTAMP
);
```

### gradients table

Computed gradients from workers:

```sql
CREATE TABLE gradients (
    id SERIAL PRIMARY KEY,
    work_unit_id INTEGER REFERENCES work_units(id),
    worker_id VARCHAR(100),
    generator_gradients BYTEA,     -- Serialized gradient tensor
    discriminator_gradients BYTEA,
    uploaded_at TIMESTAMP
);
```

### workers table

Worker registration and stats:

```sql
CREATE TABLE workers (
    worker_id VARCHAR(100) PRIMARY KEY,
    hostname VARCHAR(255),
    gpu_name VARCHAR(255),
    total_work_units INTEGER DEFAULT 0,
    total_images INTEGER DEFAULT 0,
    first_seen TIMESTAMP,
    last_heartbeat TIMESTAMP
);
```

## Coordination mechanism

### Atomic work claiming

Workers use PostgreSQL's `FOR UPDATE SKIP LOCKED`:

```sql
SELECT id FROM work_units
WHERE status = 'pending'
  AND iteration = CURRENT_ITERATION
ORDER BY id
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

This ensures:
- Only one worker claims each work unit
- No race conditions
- Failed workers don't block others

### Timeout and reclamation

Stalled work units are automatically reclaimed:

```python
# In coordinator
timeout = timedelta(minutes=5)
reclaim_query = """
    UPDATE work_units
    SET status = 'pending', claimed_by = NULL
    WHERE status = 'in_progress'
      AND claimed_at < NOW() - %s
"""
cursor.execute(reclaim_query, (timeout,))
```

### Heartbeat monitoring

Workers send periodic heartbeats:

```python
# In worker
UPDATE workers
SET last_heartbeat = NOW()
WHERE worker_id = %s
```

Coordinator can identify inactive workers:

```sql
SELECT * FROM workers
WHERE last_heartbeat < NOW() - INTERVAL '2 minutes'
```

## Communication patterns

### Pull-based architecture

Workers **pull** work from database (not pushed):
- Workers poll for available work
- No need for coordinator to track worker addresses
- Workers can join/leave anytime
- Automatically fault-tolerant

### Stateless workers

Workers don't maintain state between work units:
- Each work unit is independent
- Workers can crash and restart safely
- Easy to scale horizontally

### Centralized coordination

Database provides:
- Single source of truth
- Atomic operations
- Persistent state
- Simple debugging

## Fault tolerance

### Worker failures

- Work unit times out
- Coordinator reclaims and reassigns
- No data loss
- Training continues

### Coordinator failures

- Training state persisted in database
- Restart coordinator with `--resume`
- Picks up from last iteration
- Workers continue normally

### Database failures

- Use database replication
- Regular backups
- Minimal downtime with proper setup

## Scalability

### Horizontal scaling

Add more workers:
- No code changes needed
- Linear speedup (mostly)
- Limited by database throughput

### Vertical scaling

More powerful coordinator:
- Faster gradient aggregation
- More workers per update
- Better sample generation

### Database optimization

- Index critical columns
- Partition large tables
- Use connection pooling
- Archive old data

## Security considerations

### Database access

- Use individual student accounts
- Grant minimum necessary permissions
- No DDL access for students
- Monitor for abuse

### Data validation

- Check gradient shapes and values
- Detect outliers
- Limit upload sizes
- Rate limiting

## Next steps

- [Database schema details](database.md)
- [Coordinator internals](coordinator.md)
- [Worker implementation](worker.md)
- [Model architecture](models.md)
