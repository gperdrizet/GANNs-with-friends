# Performance tips

Optimize distributed GAN training for better speed and efficiency.

## Worker optimization

### Hardware utilization

**GPU workers:**
```yaml
# Maximize batch size for your GPU
training:
  batch_size: 128  # Adjust based on VRAM

# More DataLoader workers
worker:
  num_workers_dataloader: 8  # Use available CPU cores
```

Check GPU utilization:
```bash
watch -n 1 nvidia-smi
# GPU usage should be >90%
# Memory should be well utilized
```

**CPU workers:**
System automatically reduces batch size, but you can tune:
```yaml
training:
  batch_size: 16  # Higher if you have powerful CPU
  
worker:
  num_workers_dataloader: 0  # Avoid overhead on CPU
```

### Network optimization

**Reduce polling frequency:**
```yaml
worker:
  poll_interval: 10  # Check less often when many workers
```

**Local database cache** (for very slow networks):
```python
# Cache model weights locally
# Only download when iteration changes
if current_iteration != last_iteration:
    weights = download_weights()
    last_iteration = current_iteration
```

## Coordinator optimization

### Batch work efficiently

```yaml
training:
  batches_per_work_unit: 15  # Larger units = less overhead
  num_workers_per_update: 5  # More gradients per update
```

Trade-off: larger values = better efficiency but slower feedback.

### Parallel sample generation

Generate samples in background:
```python
import threading

def generate_samples_async():
    threading.Thread(target=generate_and_save_samples).start()
```

### Database connection pooling

```python
from psycopg2 import pool

# Create connection pool
connection_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    **db_config
)
```

## Database optimization

### Add indexes

```sql
-- Speed up work unit queries
CREATE INDEX idx_work_units_status_iteration 
ON work_units(status, iteration) WHERE status = 'pending';

CREATE INDEX idx_work_units_claimed 
ON work_units(claimed_at) WHERE status = 'in_progress';

-- Speed up worker queries
CREATE INDEX idx_workers_heartbeat 
ON workers(last_heartbeat);

-- Speed up gradient lookups
CREATE INDEX idx_gradients_work_unit 
ON gradients(work_unit_id);
```

### Regular maintenance

```sql
-- Run periodically
VACUUM ANALYZE work_units;
VACUUM ANALYZE gradients;
VACUUM ANALYZE workers;

-- Check table bloat
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Clean old data

```sql
-- Delete old gradients after aggregation
DELETE FROM gradients
WHERE work_unit_id IN (
    SELECT id FROM work_units
    WHERE iteration < (SELECT current_iteration FROM training_state) - 1
);

-- Archive completed work units
CREATE TABLE work_units_archive AS
SELECT * FROM work_units
WHERE iteration < CURRENT_ITERATION - 10;

DELETE FROM work_units
WHERE iteration < CURRENT_ITERATION - 10;
```

### Database configuration

In `postgresql.conf`:
```
# Increase connection limit
max_connections = 100

# Increase shared buffers
shared_buffers = 2GB

# Increase work mem for sorting
work_mem = 16MB

# Enable parallel query
max_parallel_workers_per_gather = 4
```

## Training optimization

### Learning rate scheduling

```python
# Decrease learning rate over time
if iteration % 1000 == 0:
    for param_group in optimizer_g.param_groups:
        param_group['lr'] *= 0.95
```

### Gradient accumulation

Simulate larger batches:
```python
accumulation_steps = 4

for i, batch in enumerate(batches):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed precision training

Faster on modern GPUs:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    fake_images = generator(noise)
    fake_output = discriminator(fake_images)
    loss_g = criterion(fake_output, real_labels)

scaler.scale(loss_g).backward()
scaler.step(optimizer_g)
scaler.update()
```

## Network optimization

### Gradient compression

```python
import torch

def compress_gradients(gradients):
    """Compress gradients before upload."""
    # Quantize to 16-bit
    compressed = {
        k: v.half() for k, v in gradients.items()
    }
    return compressed

def decompress_gradients(compressed):
    """Decompress after download."""
    return {
        k: v.float() for k, v in compressed.items()
    }
```

### Batch uploads

Upload multiple results together:
```python
# Instead of uploading after each work unit
results_buffer = []
results_buffer.append(current_result)

if len(results_buffer) >= 5:
    upload_batch(results_buffer)
    results_buffer.clear()
```

## Monitoring overhead

### Reduce logging

```python
# Log every N iterations, not every one
if iteration % 10 == 0:
    log_progress(stats)
```

### Async heartbeats

```python
import threading

def send_heartbeat_async():
    while running:
        update_heartbeat()
        time.sleep(30)

threading.Thread(target=send_heartbeat_async, daemon=True).start()
```

## Resource allocation

### Worker distribution

For N workers on same machine:
```bash
# Distribute across GPUs
CUDA_VISIBLE_DEVICES=0 python src/worker.py --config config1.yaml &
CUDA_VISIBLE_DEVICES=1 python src/worker.py --config config2.yaml &
```

### CPU affinity

Pin worker to specific cores:
```python
import os
os.sched_setaffinity(0, {0, 1, 2, 3})  # Use cores 0-3
```

## Benchmarking

### Measure worker performance

```python
import time

start = time.time()
gradients = compute_gradients(batch)
elapsed = time.time() - start

print(f'Gradient computation: {elapsed:.2f}s')
print(f'Images/sec: {batch_size / elapsed:.1f}')
```

### Profile code

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
process_work_unit()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Database query timing

```sql
-- Enable query timing
\timing on

-- Run query
SELECT * FROM work_units WHERE status = 'pending';

-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

## Best practices summary

1. **Right-size batches** - Max out GPU memory without OOM
2. **Index database** - Critical for large scale
3. **Clean old data** - Prevent database bloat
4. **Monitor continuously** - Catch issues early
5. **Profile before optimizing** - Measure, don't guess
6. **Balance trade-offs** - Speed vs quality vs complexity

## Performance targets

**Good performance:**
- GPU utilization >80%
- Work unit processing <30 seconds
- Database queries <100ms
- Worker throughput >100 images/second (GPU)

**If below targets:**
- Check batch sizes
- Profile bottlenecks
- Optimize database
- Review network latency

## Next steps

- [Troubleshooting](troubleshooting.md) - Fix performance issues
- [Architecture](../architecture/overview.md) - Understand system design
- [Configuration](../guides/configuration.md) - Tune hyperparameters
