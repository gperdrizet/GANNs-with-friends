# Contributing

Thank you for your interest in contributing to GANNs with friends!

## Ways to contribute

- Report bugs and issues
- Improve documentation
- Add new features
- Optimize performance
- Create tutorials and examples
- Help other users

## Getting started

### Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
cd GANNs-with-freinds
```

### Set up development environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install black pylint mypy
```

### Create a branch

```bash
git checkout -b feature/my-new-feature
```

## Development workflow

### 1. Make changes

Edit code following the project conventions (see below).

### 2. Test your changes

```bash
# Test manually by running the code
python src/worker.py --config config.yaml
# Or for local training:
python src/train_local.py --config config.yaml

# Check code style (optional)
black src/
pylint src/

# Type checking (optional)
mypy src/
```

**Note:** This project doesn't currently have automated tests. Please test your changes manually by running the code and verifying functionality.

### 3. Commit

```bash
git add .
git commit -m "Add feature: brief description"
```

Use clear commit messages:
- "Fix: bug in gradient aggregation"
- "Add: CPU auto-detection for workers"
- "Docs: update installation guide"
- "Refactor: simplify database queries"

### 4. Push and create pull request

```bash
git push origin feature/my-new-feature
```

Then create a pull request on GitHub.

## Code style

### Python conventions

- Follow PEP 8
- Use type hints
- Write docstrings for functions
- Keep functions focused and small
- Use meaningful variable names

```python
def load_model_weights(
    model_path: str,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Load model weights from checkpoint file.
    
    Args:
        model_path: Path to checkpoint file
        device: Target device for weights
    
    Returns:
        Dictionary containing model state dict
    """
    checkpoint = torch.load(model_path, map_location=device)
    return checkpoint['model_state_dict']
```

### Single quotes

Use single quotes for strings:

```python
# Good
config = load_config('config.yaml')

# Avoid
config = load_config("config.yaml")
```

### Sentence case for headers

All documentation headers use sentence case:

```markdown
# Installation guide

## Setup instructions

### Download dataset
```

### No emojis in code or docs

Keep documentation professional and accessible.

## Testing

**Note:** This project does not currently have automated tests. This is one of our highest priority contribution areas!

### Manual testing

For now, test your changes manually:

1. **Test worker functionality:**
   ```bash
   python src/worker.py --config config.yaml
   # Watch for errors, verify it connects to database
   ```

2. **Test coordinator:**
   ```bash
   python src/main.py --config config.yaml
   # Verify it creates work units and aggregates gradients
   ```

3. **Test local training:**
   ```bash
   python src/train_local.py --config config.yaml
   # Verify training runs without errors
   ```

4. **Check generated samples** in `data/outputs/samples/`

### Contributing automated tests

We would greatly appreciate contributions to add automated testing! See the Priority Contributions section below.

## Documentation

### Update relevant docs

If you change functionality, update:
- README.md (if user-facing)
- docs/ (detailed documentation)
- Code docstrings
- CHANGELOG.md

### Build and check docs

```bash
cd docs
pip install -r requirements.txt
make html
make serve  # View at http://localhost:8000
```

### Documentation style

- Clear and concise
- Include code examples
- Use sentence case for headers
- No emojis or symbols
- Single quotes in code examples

## Pull request guidelines

### Before submitting

- Changes tested manually (run the code and verify functionality)
- Code follows style guide
- Documentation updated
- Commits are clean and logical
- PR description explains changes

### PR description template

```markdown
## Description
Brief description of changes

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
How were changes tested manually?

## Checklist
- [ ] Tested manually (describe how)
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Reporting bugs

### Create an issue

Include:
- Clear title
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version, GPU)
- Error messages and logs

### Example bug report

```markdown
**Title**: Worker crashes when batch size > 128

**Description**:
Worker crashes with CUDA out of memory when batch_size is set above 128 in config.yaml.

**Steps to reproduce**:
1. Set batch_size: 256 in config.yaml
2. Run python src/worker.py
3. Worker crashes after claiming first work unit

**Expected**: Worker should process batch or reduce size automatically

**Actual**: RuntimeError: CUDA out of memory

**System**:
- OS: Ubuntu 22.04
- Python: 3.10.12
- GPU: NVIDIA RTX 3060 (12GB)
- PyTorch: 2.0.1+cu118

**Logs**:
[attach relevant logs]
```

## Feature requests

### Propose new features

Open an issue with:
- Clear use case
- Expected behavior
- Why it's beneficial
- Potential implementation approach

### Discuss first

For major features, discuss with maintainers before implementing.

## Code review process

### What we look for

- Correctness
- Code quality
- Test coverage
- Documentation
- Performance impact
- Backward compatibility

### Be responsive

- Respond to review comments
- Make requested changes
- Ask questions if unclear
- Be open to feedback

## Project areas

### Priority contributions

1. **Testing**
   - **Create automated test suite** (currently no tests exist!)
   - Add unit tests for models, database operations, worker logic
   - Add integration tests for full training workflows
   - Test edge cases and error handling

2. **Documentation**
   - Improve clarity
   - Add more examples
   - Create tutorials

3. **Performance**
   - Optimize database queries
   - Reduce network overhead
   - Improve gradient aggregation

4. **Features**
   - Multi-GPU support per worker
   - Gradient compression
   - Web-based monitoring dashboard
   - Support for other datasets/models

5. **Usability**
   - Better error messages
   - Improved logging
   - Setup automation

## Advanced Performance Optimization Ideas

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

## Community

### Be respectful

- Assume good intentions
- Be patient with beginners
- Give constructive feedback
- Follow code of conduct

### Help others

- Answer questions
- Review pull requests
- Improve documentation
- Share knowledge

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

- Open an issue for clarification
- Ask in pull request comments
- Contact project maintainers

## Thank you!

Your contributions make this project better for everyone. We appreciate your time and effort.
