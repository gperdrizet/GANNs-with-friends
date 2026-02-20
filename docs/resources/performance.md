# Performance Tips

Optimize your distributed GAN training setup using built-in configuration options.

## Configuration-Based Optimizations

These performance improvements work out-of-the-box by adjusting `config.yaml`.

### Batch Size Tuning

Workers can adjust batch size independently based on their hardware:

**GPU workers:**
```yaml
worker:
  batch_size: 64  # Increase for better GPU utilization (watch for OOM)
```

**Check GPU utilization in a separate terminal:**
```bash
watch -n 1 nvidia-smi
# GPU usage should be >80%
# Memory should be well utilized
```

**CPU workers:**
```yaml
worker:
  batch_size: 16  # Reduce to avoid memory issues
```

### DataLoader Workers

Adjust parallel data loading:
```yaml
data:
  num_workers_dataloader: 4  # Default: 4, adjust based on CPU cores
```

**Guidelines:**
- GPU: 4-8 workers (4 is a good default, increase if you have cores available)
- CPU: 0-2 workers (to avoid overhead)

### Work Unit Configuration

Balance database overhead vs. processing efficiency:

```yaml
training:
  images_per_work_unit: 320  # Images assigned per work unit
  num_workunits_per_update: 3  # How many work unit gradients before updating
```

**Trade-offs:**
- **Larger `images_per_work_unit`** (500-1000):
  - Less database overhead
  - Fewer work units to manage
  - Longer to process each work unit
  - Slower feedback if workers disconnect

- **Smaller `images_per_work_unit`** (100-200):
  - Faster work unit completion
  - Better for unstable workers
  - More database operations
  - Higher coordination overhead

**For `num_workunits_per_update`:**
- Set based on your expected number of workers
- Too low (1-2): Noisy gradients, potential wasted work units
- Too high (>50% of workers): Slower updates, better gradient quality
- Sweet spot: ~30-50% of your total workers

### Worker Polling

Reduce unnecessary database checks:

```yaml
worker:
  poll_interval: 5   # Seconds between work unit checks (increase if many workers)
  heartbeat_interval: 30  # Seconds between heartbeat updates
```

**When to increase poll_interval:**
- Many workers (>10): Set to 8-10 seconds
- Slow network: Set to 10-15 seconds
- Fast training iterations: Keep at 3-5 seconds

## Monitoring performance

### Check GPU Utilization

```bash
# Real-time GPU monitoring
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1

# Target: >80% GPU utilization
```

### Worker Throughput

Monitor worker output for:
- **Images/second**: Should be >100 for GPU, >10 for CPU
- **Work unit completion time**: Should be <60 seconds for typical settings
- **Gradient upload time**: Should be <5 seconds

### Database Performance

If work units take too long to claim or complete:
- Check network latency to database server
- Verify database server isn't overloaded
- Consider reducing `poll_interval`

## Best practices

1. **Start conservative** - Begin with default settings
2. **Monitor first** - Watch GPU/CPU usage before optimizing
3. **Change one thing at a time** - Easier to identify impact
4. **Match batch size to hardware** - Max out GPU memory without OOM errors
5. **Tune for your class size** - Set `num_workunits_per_update` based on worker count

## Performance targets

**Good performance indicators:**
- GPU utilization: >80%
- Work unit processing: <30 seconds (for default config)
- Worker throughput: >100 images/second (GPU), >10 images/second (CPU)
- Database query time: <100ms

**If below targets:**
- Increase batch size (GPU workers)
- Increase num_workers_dataloader (if CPU available)
- Check network connection to database
- See [Troubleshooting](troubleshooting.md)

## Example configurations

**Note:** Default settings in `config.yaml.template`:
- `images_per_work_unit: 320`
- `num_workunits_per_update: 3`
- `batch_size: 32` (in worker section)
- `num_workers_dataloader: 4`
- `poll_interval: 5`

These examples show how to adjust for different class sizes:

### Small class (3-5 workers)
```yaml
training:
  images_per_work_unit: 320
  num_workunits_per_update: 2

worker:
  batch_size: 64  # Workers tune based on their GPU
  poll_interval: 5
```

### Medium class (10-20 workers)
```yaml
training:
  images_per_work_unit: 480
  num_workunits_per_update: 8

worker:
  batch_size: 64  # Workers tune based on their GPU
  poll_interval: 8
```

### Large class (30+ workers)
```yaml
training:
  images_per_work_unit: 640
  num_workunits_per_update: 15

worker:
  batch_size: 64  # Workers tune based on their GPU
  poll_interval: 10
```

## Next steps

- [Configuration Guide](../guides/configuration.md) - Detailed config options
- [Troubleshooting](troubleshooting.md) - Fix performance issues
- [Contributing](contributing.md) - Help implement advanced optimizations
