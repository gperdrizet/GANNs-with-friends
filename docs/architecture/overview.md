# Architecture overview

Understanding the system architecture of distributed GAN training.

## System components

The system consists of four main components:

1. **Coordinator** - Main training process (instructor runs this)
2. **Workers** - Distributed compute nodes (students run these)
3. **PostgreSQL Database** - Coordination and communication hub
4. **Shared Model** - GAN being trained collaboratively

## High-level architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Coordinator (Instructor)              │
│  ┌────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Create     │  │  Aggregate  │  │   Update        │  │
│  │ Work Units │→ │  Gradients  │→ │   Weights       │  │
│  └────────────┘  └─────────────┘  └─────────────────┘  │
│         │              ↑                    │            │
└─────────┼──────────────┼────────────────────┼───────────┘
          ↓              │                    ↓
┌─────────────────────────────────────────────────────────┐
│                PostgreSQL Database                       │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐ │
│  │  Work    │  │ Gradients│  │Weights │  │ Training │ │
│  │  Units   │  │          │  │        │  │  State   │ │
│  └──────────┘  └──────────┘  └────────┘  └──────────┘ │
└─────────┬──────────────┬────────────────────┬───────────┘
          │              │                    │
┌─────────┴──────┐  ┌────┴────────┐  ┌───────┴────────┐
│  Worker 1      │  │  Worker 2   │  │  Worker N      │
│  (Student GPU) │  │ (Student    │  │  (Student CPU) │
│                │  │  GPU)       │  │                │
│  ┌──────────┐  │  │ ┌──────────┐│  │ ┌──────────┐  │
│  │ Compute  │  │  │ │ Compute  ││  │ │ Compute  │  │
│  │ Gradients│  │  │ │ Gradients││  │ │ Gradients│  │
│  └──────────┘  │  │ └──────────┘│  │ └──────────┘  │
└────────────────┘  └─────────────┘  └────────────────┘
```

## Data flow

### 1. Initialization (Coordinator)

```
Coordinator starts
├─> Initialize generator and discriminator
├─> Save initial weights to database
├─> Create work units for epoch 1
│   └─> Each work unit = batch of image indices
└─> Set training_state (iteration=0, epoch=1)
```

### 2. Worker claims and processes

```
Worker polls database
├─> Find unclaimed work unit
├─> Atomically claim it (FOR UPDATE SKIP LOCKED)
├─> Download current model weights
├─> Load assigned images from CelebA
├─> Forward pass through models
├─> Compute gradients
├─> Upload gradients to database
└─> Mark work unit as completed
```

### 3. Coordinator aggregates

```
Coordinator waits for N workers
├─> Check completed work units
├─> When N workers done:
│   ├─> Download N gradient tensors
│   ├─> Average gradients
│   ├─> Apply optimizer step
│   ├─> Update model weights
│   ├─> Save new weights to database
│   ├─> Increment iteration counter
│   └─> Create next batch of work units
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
