# Quick start

Get up and running with distributed GAN training in minutes.

## For students (workers)

### Step 1: Choose your path

Pick the installation method that works best for you:

- **Easiest**: [Google Colab](../setup/google-colab.md) - No installation needed
- **Full features**: [Dev container](../setup/dev-container.md) - Complete development environment
- **Direct**: [Native Python](../setup/native-python.md) - Install directly on your system
- **Conda users**: [Conda environment](../setup/conda.md) - Use conda package manager

### Step 2: Get database credentials

Contact your instructor to receive:
- Database host address
- Database name
- Username
- Password

### Step 3: Configure and run

All paths follow the same basic pattern:

1. Clone or fork the repository
2. Install dependencies (varies by path)
3. Download the CelebA dataset
4. Configure database connection in `config.yaml`
5. Start the worker: `python src/worker.py`

Your GPU (or CPU) is now part of the training cluster!

## For instructors (coordinator)

### Step 1: Set up database

Deploy a publicly accessible PostgreSQL database:

```bash
# Example using PostgreSQL
createdb distributed_gan
psql distributed_gan < src/database/schema.sql
```

Or use a cloud provider:
- AWS RDS
- Google Cloud SQL
- Azure Database for PostgreSQL
- ElephantSQL (free tier available)

### Step 2: Initialize the training system

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_celeba.py

# Configure database in config.yaml
cp config.yaml.template config.yaml
# Edit config.yaml with your database details

# Initialize database schema
python src/database/init_db.py
```

### Step 3: Start coordinator

```bash
python src/main.py --epochs 50 --sample-interval 1
```

The coordinator will:
- Create work units for workers to claim
- Aggregate gradients from completed work
- Update model weights
- Generate sample images periodically
- Save checkpoints

### Step 4: Monitor progress

Watch the output for:
- Number of active workers
- Training iteration progress
- Loss values
- Sample generation

Check generated samples in `data/outputs/samples/`.

## Optional: Hugging Face integration

Enable automatic model uploads to share progress with students:

```yaml
# In config.yaml
huggingface:
  enabled: true
  repo_id: your-username/distributed-gan-celeba
  token: your_hf_token
  push_interval: 5
```

Students can then view live results without running the training themselves.

## Viewing results

After training starts, use the demo notebook:

```bash
jupyter notebook notebooks/demo_trained_model.ipynb
```

The notebook will:
- Download the latest model from Hugging Face (or use local checkpoint)
- Generate new celebrity faces
- Show training progress

## Troubleshooting

**No work units available:**
- Wait for the coordinator to start and create work units
- Check database connection

**Worker crashes:**
- Reduce batch size in `config.yaml`
- Check GPU memory with `nvidia-smi`
- Try CPU-only mode

**Slow training:**
- Need more workers participating
- Check network connection to database
- Verify workers are actively completing work units

See the [troubleshooting guide](../resources/troubleshooting.md) for more help.

## Next steps

- [Student guide](../guides/students.md) - Detailed student workflow
- [Instructor guide](../guides/instructors.md) - Coordinator management
- [Configuration reference](../guides/configuration.md) - All config options
