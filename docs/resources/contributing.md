# Contributing

Thank you for your interest in contributing to GANNs with friends!

## Ways to contribute

- Report bugs and issues
- Improve documentation
- Add new features
- Optimize performance
- Create tutorials and examples

## Getting started

### Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
cd GANNs-with-freinds
git checkout -b feature/my-new-feature
```

### Make your changes

1. Edit the code
2. Test manually by running the affected scripts
3. Update documentation if needed
4. Commit with a clear message
5. Push and create a pull request

## Pull request guidelines

- Test your changes manually
- Update documentation if needed
- Write clear commit messages
- Describe what changed and why in the PR description

## Reporting bugs

Include in your bug report:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- System info (OS, Python version, GPU)
- Error messages and logs

## Development philosophy

This project uses AI-assisted development with human oversight. For details about the development approach, collaborative workflow, and concrete examples, see the [Development approach](../development-approach.md) section.

## Advanced performance optimization ideas

These are advanced code modifications that could significantly improve system performance. They require deeper understanding of the codebase and are excellent contributions for experienced developers or students learning about distributed systems optimization.

### Database Optimizations

#### Add indexes for faster queries

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

#### Connection pooling

Currently each database operation creates a new connection. Implementing connection pooling could reduce overhead:

```python
from psycopg2 import pool

class DatabaseManager:
    def __init__(self):
        self.connection_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            **db_config
        )
    
    def get_connection(self):
        return self.connection_pool.getconn()
    
    def release_connection(self, conn):
        self.connection_pool.putconn(conn)
```

#### Regular maintenance

Add automated database cleanup:

```sql
-- Delete old gradients after aggregation
DELETE FROM gradients
WHERE work_unit_id IN (
    SELECT id FROM work_units
    WHERE iteration < (SELECT current_iteration FROM training_state) - 1
);

-- Archive old work units
CREATE TABLE work_units_archive AS
SELECT * FROM work_units
WHERE iteration < CURRENT_ITERATION - 10;

DELETE FROM work_units
WHERE iteration < CURRENT_ITERATION - 10;
```

### Training Optimizations

#### Mixed precision training

Use automatic mixed precision for faster training on modern GPUs:

```python
from torch.cuda.amp import autocast, GradScaler

class Worker:
    def __init__(self, config_path):
        # ... existing init code ...
        self.scaler = GradScaler()
    
    def compute_gradients(self, real_batch):
        with autocast():
            fake_images = self.generator(self.noise)
            fake_output = self.discriminator(fake_images)
            loss_g = self.criterion(fake_output, self.real_labels)
        
        self.scaler.scale(loss_g).backward()
        # ... rest of gradient computation
```

#### Gradient accumulation

Simulate larger batch sizes without increasing memory:

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

### Network Optimizations

#### Gradient compression

Reduce network traffic by compressing gradients:

```python
def compress_gradients(gradients):
    """Compress gradients before upload to database."""
    # Quantize to float16
    compressed = {
        k: v.half() for k, v in gradients.items()
    }
    # Could also use: top-k sparsification, quantization, etc.
    return compressed

def decompress_gradients(compressed):
    """Decompress after download from database."""
    return {
        k: v.float() for k, v in compressed.items()
    }
```

#### Batched uploads

Upload multiple work unit results together:

```python
class Worker:
    def __init__(self):
        self.results_buffer = []
        self.buffer_size = 5
    
    def process_work_unit(self, work_unit):
        gradients = self.compute_gradients(work_unit)
        self.results_buffer.append((work_unit.id, gradients))
        
        if len(self.results_buffer) >= self.buffer_size:
            self.upload_batch(self.results_buffer)
            self.results_buffer.clear()
```

#### Local weight caching

Only download weights when they change:

```python
class Worker:
    def __init__(self):
        self.cached_iteration = -1
        self.cached_weights = None
    
    def get_weights(self):
        current_iteration = self.db.get_current_iteration()
        if current_iteration != self.cached_iteration:
            self.cached_weights = self.db.get_weights()
            self.cached_iteration = current_iteration
        return self.cached_weights
```

### Monitoring Optimizations

#### Async heartbeats

Send heartbeats in background thread to avoid blocking:

```python
import threading

class Worker:
    def start_heartbeat(self):
        def heartbeat_loop():
            while self.running:
                self.db.update_heartbeat(self.worker_id)
                time.sleep(30)
        
        self.heartbeat_thread = threading.Thread(
            target=heartbeat_loop, 
            daemon=True
        )
        self.heartbeat_thread.start()
```

#### Reduced logging overhead

Log less frequently in tight loops:

```python
# Instead of logging every iteration
if iteration % 10 == 0:
    logger.info(f"Progress: {iteration}")
```

### Profiling Tools

#### Measure performance

Add benchmarking code to find bottlenecks:

```python
import time

class PerformanceTimer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"{self.name}: {elapsed:.3f}s")

# Usage:
with PerformanceTimer("Gradient computation"):
    gradients = compute_gradients(batch)

with PerformanceTimer("Database upload"):
    upload_gradients(gradients)
```

#### Code profiling

Use cProfile to find slow functions:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
process_work_unit()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Show top 20 functions
```

### Resource Management

#### CPU affinity

Pin workers to specific CPU cores:

```python
import os

def set_cpu_affinity(core_ids):
    """Pin process to specific CPU cores."""
    os.sched_setaffinity(0, set(core_ids))

# Example: use cores 0-3 for this worker
set_cpu_affinity([0, 1, 2, 3])
```

#### Multiple workers per GPU

Run multiple worker processes on one GPU:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python src/worker.py --config config.yaml &

# Terminal 2
CUDA_VISIBLE_DEVICES=0 python src/worker.py --config config.yaml &
```

### Implementation Notes

**Before implementing:**
- Profile to confirm the optimization is needed
- Understand the trade-offs (complexity vs. performance gain)
- Test manually to verify functionality
- Document the changes thoroughly
- Consider backward compatibility

**Testing performance improvements:**
- Measure before and after with realistic workloads
- Test with multiple workers, not just one
- Check for edge cases and failure modes
- Verify results are numerically equivalent

**Good first optimizations:**
1. Database indexes (easy, high impact)
2. Local weight caching (medium difficulty, good gains)
3. Reduced logging (easy, modest gains)
4. Connection pooling (medium difficulty, good for many workers)

**Advanced optimizations:**
1. Mixed precision training (requires careful testing)
2. Gradient compression (complex, measure quality impact)
3. Async operations (increases code complexity)

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

Open an issue or contact the project maintainers.
