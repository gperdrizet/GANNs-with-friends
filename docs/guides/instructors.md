# Instructor guide

Guide for instructors coordinating distributed GAN training.

## Your role

As the instructor/coordinator, you:
- Set up and manage the PostgreSQL database
- Run the main coordinator process
- Monitor worker participation
- Track training progress
- Share results with students

## Pre-training setup

### 1. Deploy database

Choose a cloud database provider:

**Recommended options:**
- **ElephantSQL** - Free tier available, easy setup
- **AWS RDS** - Reliable, scalable
- **Google Cloud SQL** - Good integration with Colab
- **Azure Database for PostgreSQL** - Enterprise features
- **Self-hosted** - Full control, requires server

**Database requirements:**
- PostgreSQL 12 or later
- Publicly accessible
- At least 1GB storage
- Support for BLOB storage

### 2. Initialize database

```bash
# Create database
createdb distributed_gan

# Initialize schema
python src/database/init_db.py --config config.yaml
```

This creates tables:
- `training_state` - Current iteration, epoch, weights
- `work_units` - Individual batch assignments
- `workers` - Worker registration and stats
- `gradients` - Uploaded gradient arrays

### 3. Create student accounts

For security, create individual accounts:

```sql
-- Create user
CREATE USER student1 WITH PASSWORD 'secure_password';

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO student1;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO student1;
```

Distribute credentials securely (email, LMS, etc.).

### 4. Configure coordinator

Edit `config.yaml`:

```yaml
database:
  host: your-database.provider.com
  port: 5432
  database: distributed_gan
  user: coordinator  # Your admin account
  password: your_secure_password

training:
  batch_size: 32
  batches_per_work_unit: 10 
  num_workunits_per_update: 3  # Wait for N work units before updating

huggingface:  # Optional but recommended for your own training runs
  enabled: true
  repo_id: your-username/your-repo  # Create at huggingface.co/new
  token: your_hf_write_token        # From huggingface.co/settings/tokens
  push_interval: 5
```

## Running training

### Start coordinator

```bash
python src/main.py --epochs 50 --sample-interval 1
```

The coordinator will:
- Initialize model weights
- Create work units for epoch 1
- Wait for workers to claim and complete units
- Aggregate gradients when enough workers finish
- Update models and create next batch of work
- Generate sample images periodically
- Push to Hugging Face (if enabled)

### Monitor progress

**Console output:**
```
Initializing coordinator...
Database initialized
Created 1,582 work units for epoch 1

Waiting for workers... (0/3 completed)
Worker abc123 completed work unit 1
Worker def456 completed work unit 2
Worker ghi789 completed work unit 3

Aggregating gradients from 3 workers...
Applied gradient update
Generator loss: 2.345
Discriminator loss: 1.234

Generated samples saved to data/outputs/samples/iteration_0001.png
Pushed checkpoint to Hugging Face

Creating work units for iteration 2...
```

**Database queries:**

```sql
-- Check active workers
SELECT worker_id, gpu_name, total_work_units, last_heartbeat
FROM workers
WHERE last_heartbeat > NOW() - INTERVAL '2 minutes'
ORDER BY total_work_units DESC;

-- Training progress
SELECT current_iteration, current_epoch, 
       generator_loss, discriminator_loss
FROM training_state;

-- Work unit status
SELECT status, COUNT(*)
FROM work_units
WHERE iteration = (SELECT current_iteration FROM training_state)
GROUP BY status;
```

## Managing the training session

### Pause training

Press `Ctrl+C` - coordinator stops gracefully after current iteration.

### Resume training

Just restart:
```bash
python src/main.py --resume
```

Loads latest state from database and continues.

### Adjust parameters mid-training

Edit `config.yaml` and restart coordinator. Changes take effect:
- `batch_size` - affects new work units
- `num_workunits_per_update` - how many work units to wait for
- `sample_interval` - frequency of sample generation

### Handle stalled workers

Workers that crash leave work units as "in_progress". The coordinator automatically reclaims units after timeout (default 5 minutes).

Manual reclaim:
```sql
UPDATE work_units
SET status = 'pending', claimed_by = NULL
WHERE status = 'in_progress' 
  AND claimed_at < NOW() - INTERVAL '10 minutes';
```

## Monitoring tools

### Real-time visualization

Create a simple dashboard:

```python
# monitor.py
import psycopg2
import time

while True:
    # Connect and query
    stats = get_training_stats()
    print(f"Iteration: {stats['iteration']}")
    print(f"Active workers: {stats['active_workers']}")
    print(f"Work units completed: {stats['completed']}/{stats['total']}")
    print(f"Estimated time remaining: {stats['eta']}")
    time.sleep(10)
```

### Generated samples

Check `data/outputs/samples/` for periodic image samples. Share with class to show progress.

### Hugging Face integration

If enabled, students can:
- View latest model on Hugging Face
- Download and generate faces
- See training progress in real-time

## Best practices

### Before class

- Test full workflow yourself
- Verify database is accessible from various networks
- Prepare student credentials in advance
- Set up Hugging Face repo (optional)
- Create demo notebook showing results

### During class

- Start coordinator before students join
- Monitor database for first few students
- Be available for setup troubleshooting
- Share generated samples periodically
- Track who's participating

### After class

- Save final checkpoint
- Export worker statistics
- Create visualizations of results
- Share model on Hugging Face
- Get student feedback

## Troubleshooting

### No workers connecting

- Verify database is publicly accessible
- Check firewall rules
- Test with your own worker
- Verify student credentials

### Training very slow

- Need more workers (encourage participation)
- Increase `num_workunits_per_update` to wait for more work unit gradients
- Check database performance
- Verify network isn't bottleneck

### Unstable training

- GAN training is inherently unstable
- Try lowering learning rates
- Check for bad gradients from workers
- May need to restart and adjust hyperparameters

### Database full

- Gradients table grows large
- Add cleanup: delete old gradients after aggregation
- Increase database storage
- Archive old iterations

## Grading and assessment

### Track individual contributions

```sql
SELECT worker_id, 
       COUNT(*) as work_units,
       SUM(processing_time) as total_time,
       MIN(completed_at) as first_contribution,
       MAX(completed_at) as last_contribution
FROM work_units
WHERE status = 'completed'
GROUP BY worker_id
ORDER BY work_units DESC;
```

### Metrics to consider

- Number of work units completed
- Total processing time contributed
- Consistency (spread over time vs burst)
- Quality (check for errors)

### Export results

```bash
# Save statistics
psql -h HOST -U USER -d DATABASE -c \
  "COPY (SELECT * FROM workers) TO STDOUT CSV HEADER" \
  > worker_stats.csv
```

## Advanced topics

### Multiple coordinator instances

For very large classes, run multiple coordinators:
- Each handles different epoch ranges
- Coordinate via database flags
- Requires careful synchronization

### Custom work unit creation

Modify work unit generation:
- Stratified sampling
- Hard example mining
- Progressive difficulty

### Gradient verification

Add checks to detect malicious/broken workers:
- Gradient magnitude limits
- Statistical outlier detection
- Comparison across workers

## Learning objectives

This project teaches students:

**Technical skills:**
- Distributed system architecture
- Database-coordinated computing
- GAN training dynamics
- Python development

**Soft skills:**
- Collaboration at scale
- Troubleshooting
- Reading documentation
- Contributing to shared goals

## Next steps

- [Configuration reference](configuration.md) - All config options
- [Monitoring guide](monitoring.md) - Advanced monitoring
- [Architecture overview](../architecture/coordinator.md) - How coordinator works
