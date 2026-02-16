# Quick Reference Guide

## For students (workers)

### Initial setup
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
cd GANNs-with-freinds

# 2. Open in VS Code Dev Container
# Click "Reopen in Container" when prompted

# 3. Install dependencies and setup
pip install -r requirements.txt
cp config/config.yaml.template config/config.yaml

# Edit config/config.yaml with database credentials
python scripts/download_celeba.py
```

### Starting your worker
```bash
cd src
python worker.py
```

### Stopping your worker
Press `Ctrl+C` in the terminal

### Troubleshooting
```bash
# Test GPU
python -c "import torch; print(torch.cuda.is_available())"

# Test database connection
cd src/database
python init_db.py

# Test dataset
cd src/data
python dataset.py ../../data/celeba

# Check worker logs
# Look for "Processing work unit..." messages
```

## Local training (single GPU)

### Quick start
```bash
cd src
python train_local.py --epochs 50 --batch-size 128
```

### Common commands
```bash
# Train with custom settings
python train_local.py --epochs 100 --batch-size 64 --lr 0.0001

# Resume from checkpoint
python train_local.py --resume outputs_local/checkpoints/checkpoint_latest.pth

# Train with smaller batches (for low VRAM)
python train_local.py --batch-size 32

# Quick test run (1 epoch)
python train_local.py --epochs 1 --sample-interval 1
```

### Monitoring
```bash
# Watch for new samples
ls -ltr outputs_local/samples/

# Check checkpoints
ls -ltr outputs_local/checkpoints/
```

### Quick test
```bash
# Test local training (1 epoch)
cd src
python train_local.py --epochs 1 --batch-size 64
```

### Performance comparison
```bash
# Distributed: Multiple GPUs working together
# - Slower per-iteration (network overhead)
# - Educational: Learn distributed systems

# Local: Single GPU standard training  
# - Faster per-iteration (no network)
# - Baseline for comparison
# - Good for experimentation
```

## Viewing results

### Demo notebook
```bash
# After training, visualize results with the demo notebook
jupyter notebook notebooks/demo_trained_model.ipynb

# Or use JupyterLab
jupyter lab notebooks/demo_trained_model.ipynb
```

**What the notebook shows:**
- Training loss curves (G and D losses)
- Grid of generated face images
- Training progression over epochs
- Interactive face generation

### Quick view
```bash
# Just look at sample images
ls -ltr outputs_local/samples/  # Local training
ls -ltr outputs/samples/        # Distributed training

# View latest sample
xdg-open outputs_local/samples/epoch_*.png  # Linux
```

## For instructor (main coordinator)

### First time setup
```bash
# 1. Setup PostgreSQL database (containerized)
# Note connection details

# 2. Follow student setup steps

# 3. Initialize database
cd src
python database/init_db.py

# 4. Create initial model weights
python main.py --epochs 1 --sample-interval 1
# Stop it after first iteration (Ctrl+C)
```

### Starting training
```bash
cd src
python main.py --epochs 50 --sample-interval 1
```

### Hugging Face Hub integration (optional)
```bash
# 1. Install if not already installed
pip install huggingface-hub

# 2. Create a model repo at https://huggingface.co/new

# 3. Get your token from https://huggingface.co/settings/tokens

# 4. Edit config/config.yaml:
# huggingface:
#   enabled: true
#   repo_id: your-username/distributed-gan-celeba
#   token: YOUR_HF_TOKEN
#   push_interval: 5

# 5. Start training - models will auto-push to Hugging Face

# Students can then view progress in real-time by:
# - Opening notebooks/demo_trained_model.ipynb
# - Setting USE_HUGGINGFACE = True
# - Setting HF_REPO_ID = 'your-username/distributed-gan-celeba'
```

### Monitoring
```bash
# Watch outputs/samples/ directory for generated images
ls -ltr outputs/samples/

# Check active workers
# Query database: SELECT * FROM workers WHERE last_heartbeat > NOW() - INTERVAL '2 minutes';

# Check progress
# Query database: SELECT * FROM training_state;
```

### Stopping training
Press `Ctrl+C` in the terminal. Workers will automatically stop when they detect training is inactive.

### Resetting training
```bash
cd src
python database/init_db.py --reset
```

## Database queries

### Check training progress
```sql
SELECT current_iteration, current_epoch, 
       g_loss, d_loss, 
       total_images_processed,
       training_active
FROM training_state;
```

### List active workers
```sql
SELECT worker_id, hostname, gpu_name, 
       total_work_units, total_images,
       last_heartbeat
FROM workers
WHERE last_heartbeat > NOW() - INTERVAL '2 minutes'
ORDER BY total_images DESC;
```

### Work unit status
```sql
SELECT iteration, status, COUNT(*) as count
FROM work_units
GROUP BY iteration, status
ORDER BY iteration DESC, status;
```

### Top contributors
```sql
SELECT worker_id, gpu_name, 
       total_work_units, 
       total_batches,
       total_images
FROM workers
ORDER BY total_images DESC
LIMIT 10;
```

## File structure

```
Important files:
├── config/config.yaml          # YOUR credentials (don't commit!)
├── src/
│   ├── worker.py               # Students run this
│   ├── main.py                 # Instructor runs this
│   └── database/init_db.py     # Database setup
├── data/celeba/                # Dataset location
└── outputs/samples/            # Generated images appear here
```

## Common issues

| Issue | Solution |
|-------|----------|
| Can't connect to DB | Check config.yaml credentials |
| Out of memory | Reduce batch_size in config.yaml or command line |
| No work available | Wait for main process to create work units |
| Worker not updating | Check heartbeat in database |
| Slow training | Need more workers or larger batches |
| Local training OOM | Use smaller batch size: `--batch-size 32` |
| Local training slow | Reduce num_workers: `--num-workers 2` |

## Performance tuning

### Distributed training - if you have low VRAM (4-6GB)
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Distributed training - if you have high VRAM (12GB+)
```yaml
training:
  batch_size: 64  # Increase from 32
```

### Local training - batch size guide
```bash
# Low VRAM (4-6GB)
python train_local.py --batch-size 32

# Medium VRAM (8-10GB)  
python train_local.py --batch-size 64

# High VRAM (12GB+)
python train_local.py --batch-size 128

# Very High VRAM (24GB+)
python train_local.py --batch-size 256
```

### If database is slow
```yaml
training:
  batches_per_work_unit: 20  # Increase from 10
  num_workers_per_update: 5  # Increase from 3
```

## Expected timeline

- **Setup:** 15-30 minutes (dataset download takes longest)
- **Training:** 4-5 hours with 10 workers
- **Results:** Recognizable faces after ~2 hours

## Tips for success

### Students
- Keep your worker running for the entire session
- Don't change config during training
- Close other GPU applications
- Check the samples directory to see progress

### Instructor
- Start training before students arrive
- Have database credentials ready to share
- Monitor for offline workers
- Save interesting sample generations
- Take screenshots for demonstration

## Emergency procedures

### Worker crashed
Just restart it - it will automatically pick up where it left off

### Main process crashed
Restart it - it will resume from last saved iteration

### Database connection lost
Workers will retry automatically. Check database availability.

### Need to stop everything
1. Stop main process (Ctrl+C)
2. Workers will stop automatically when they detect inactive training
3. OR manually stop database to force all connections closed
