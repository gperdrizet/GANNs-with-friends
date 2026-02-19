# Student guide

Guide for students participating in distributed GAN training as workers.

## Your role

As a worker, you:
- Contribute GPU/CPU processing power
- Compute gradients on assigned image batches
- Help train the shared GAN model
- Learn about distributed systems and GANs

You don't need to understand every detail - the worker handles most complexity automatically.

## Getting started

### 1. Choose your setup path

Pick what works best for you:

- [Google Colab](../setup/google-colab.md) - Easiest, no local install
- [Dev container](../setup/dev-container.md) - Full development environment
- [Native Python](../setup/native-python.md) - Direct local installation
- [Conda](../setup/conda.md) - If you use conda

### 2. Get credentials

Your instructor will provide:
- Database host address
- Database name
- Your username
- Your password

Keep these secure - don't share publicly.

### 3. Configure and run

After installing dependencies:

```bash
# Copy template
cp config.yaml.template config.yaml

# Edit with your credentials
nano config.yaml  # or use any text editor

# Start contributing!
python src/worker.py
```

### 4. Set your name (optional)

Add your name to `config.yaml` so you appear on the dashboard leaderboard:

```yaml
worker:
  name: Alice  # Your name here!
  poll_interval: 5
```

This makes it easy to track your contributions and compete with classmates!

## What the worker does

### Automatic workflow

The worker runs in a loop:

1. **Poll database** - Check for available work units
2. **Claim work** - Atomically claim an unclaimed batch
3. **Load weights** - Get latest model weights
4. **Compute gradients** - Process assigned images
5. **Upload results** - Send gradients to database
6. **Repeat** - Continue until training completes

You just start it and let it run!

### What you'll see

```
Initializing worker...
Dataset found locally (202,599 images)
Worker abc123 initialized successfully!
Name: Alice
GPU: NVIDIA GeForce RTX 3080
Polling for work...

Processing work unit 42 (iteration 5)...
Loaded generator weights (iteration 5)
Loaded discriminator weights (iteration 5)
Processing batch 1/10...
Processing batch 2/10...
...
Completed work unit 42 in 15.3s

Processing work unit 43 (iteration 5)...
...
```

## Understanding the output

### Initialization

```
Worker abc123 initialized successfully!
Name: Alice
GPU: NVIDIA GeForce RTX 3080
Dataset found locally (202,599 images)
```

- **Worker ID**: Unique identifier for your worker
- **Name**: Your name (from config, or hostname if not set)
- **GPU**: Your hardware (or "CPU" if no GPU)
- **Dataset**: Shows if found locally or downloaded

### During training

```
Processing work unit 42 (iteration 5)...
Completed work unit 42 in 15.3s
Processed 320 images total
```

- **Work unit**: Unique batch assignment
- **Iteration**: Current training iteration
- **Time**: How long processing took
- **Total images**: Cumulative contribution

### Heartbeat

```
Heartbeat sent
```

Periodic check-in to show you're still active.

## Monitoring your contribution

### In the console

The worker prints:
- Number of work units completed
- Total images processed
- Processing time per unit

### Database queries

Your instructor can show your stats:

```sql
SELECT worker_id, total_images, total_work_units, last_heartbeat
FROM workers
WHERE worker_id = 'YOUR_WORKER_ID';
```

### Leaderboard (if available)

Your instructor may set up a dashboard showing:
- Top contributors by name
- Total work units processed
- Active workers

Run it yourself with `streamlit run src/dashboard.py`

## Best practices

### Maximize contribution

- **Let it run** - Keep worker running as long as possible
- **Stable connection** - Ensure reliable internet
- **Avoid interruptions** - Close unnecessary applications
- **Monitor occasionally** - Check it hasn't crashed

### Resource management

- **GPU memory** - Close other GPU applications
- **CPU usage** - Worker uses one CPU core mostly
- **Disk space** - Needs ~10GB for dataset
- **Network** - Upload/download gradients and weights

### When to stop

You can stop anytime:
- Press **Ctrl+C** in terminal
- Worker will finish current work unit gracefully
- No data loss - training state is in database

## Troubleshooting

### No work units available

**Problem**: Worker keeps polling but finds no work

**Solutions**:
- Training may not have started yet
- Wait for coordinator to create work units
- Check with instructor

### Connection errors

**Problem**: Can't connect to database

**Solutions**:
- Verify credentials in `config.yaml`
- Check database host is accessible
- Test connection: `ping DATABASE_HOST`
- Contact instructor

### Out of memory

**Problem**: GPU runs out of memory

**Solutions**:
- Close other GPU applications
- Modify `config.yaml`: reduce `batch_size`
- Try CPU-only mode

### Worker crashes

**Problem**: Worker stops unexpectedly

**Solutions**:
- Check error message
- Verify dataset downloaded completely
- Try reducing batch size
- Restart worker - it will resume automatically

### Slow performance

**Problem**: Processing work units very slowly

**Solutions**:
- Check GPU utilization: `nvidia-smi`
- Verify using GPU not CPU
- Close background applications
- Check network speed

## FAQ

**Q: How long should I run the worker?**  
A: As long as you can! Even 30 minutes helps. Ideally several hours.

**Q: Will this harm my GPU?**  
A: No. GPUs are designed for this. Monitor temperature if concerned (should stay under 85°C).

**Q: Can I use my computer while the worker runs?**  
A: Yes, but close other GPU applications. CPU work is fine.

**Q: What if I need to stop?**  
A: Just press Ctrl+C. Worker stops gracefully. You can restart anytime.

**Q: Do I get credit for contribution?**  
A: Your instructor tracks contributions. Check your course requirements.

**Q: Can I run multiple workers?**  
A: Yes, if you have multiple GPUs. Each needs separate config.

**Q: What if training finishes?**  
A: Worker will detect completion and stop. Check with instructor.

## Learning outcomes

By participating as a worker, you learn:

**Distributed systems:**
- How workers coordinate without direct communication
- Database as message queue
- Atomic operations and race conditions
- Fault tolerance

**Deep learning:**
- GAN architecture
- Gradient computation
- Data parallel training
- The role of batch processing

**Practical skills:**
- Environment setup
- Configuration management
- Monitoring processes
- Debugging distributed applications

## Next steps

- [View results](../features/huggingface-integration.md) - See the trained model
- [Architecture](../architecture/worker.md) - Understand worker internals
- [Local training](../setup/local-training.md) - Train your own model
