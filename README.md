# Distributed GAN training with students as workers

An educational distributed deep learning system where students become part of a compute cluster to train a GAN (Generative Adversarial Network) to generate celebrity faces.

## Documentation

**[📚 Full documentation](https://gperdrizet.github.io/GANNs-with-freinds/)** (coming soon)

For detailed guides, visit the docs:
- [Getting started](docs/getting-started/overview.md)
- [Installation options](docs/getting-started/installation.md)
- [Student guide](docs/guides/students.md)
- [Instructor guide](docs/guides/instructors.md)
- [Architecture overview](docs/architecture/overview.md)

Build docs locally:
```bash
cd docs
pip install -r requirements.txt
make html
make serve  # View at http://localhost:8000
```

## Concept

This project demonstrates distributed machine learning by:
- Using students' GPUs as a distributed compute cluster
- Coordinating training through a PostgreSQL database (no complex networking!)
- Training a DCGAN to generate realistic face images
- Teaching distributed systems, parallel training, and GANs simultaneously

## Architecture

**Main process (instructor):**
- Creates work units (batches of image indices)
- Aggregates gradients from workers
- Applies optimizer steps
- Tracks training progress

**Worker process (students):**
- Polls database for available work
- Computes gradients on assigned image batches
- Uploads gradients back to database
- Runs continuously until training completes

**PostgreSQL database:**
- Stores model weights, gradients, work units
- Acts as communication hub (no port forwarding needed!)
- Tracks worker statistics for monitoring

## Project structure

```
GANNs-with-freinds/
├── src/
│   ├── models/
│   │   └── dcgan.py              # Generator and discriminator models
│   ├── data/
│   │   └── dataset.py            # CelebA dataset loader
│   ├── database/
│   │   ├── schema.py             # Database table definitions
│   │   ├── db_manager.py         # Database operations
│   │   └── init_db.py            # Database initialization
│   ├── worker.py                 # Worker process (students run this)
│   ├── main.py                   # Main coordinator (instructor runs this)
│   ├── train_local.py            # Local single-GPU training (no database)
│   └── utils.py                  # Helper functions
├── scripts/
│   └── download_celeba.py        # Dataset download script
├── config.yaml.template          # Configuration template
├── notebooks/
│   └── demo_trained_model.ipynb  # Demo notebook for visualizing results
├── data/                         # CelebA dataset goes here
├── outputs/
│   ├── samples/                  # Generated image samples
│   └── checkpoints/              # Model checkpoints
├── DESIGN.md                     # Detailed design document
├── requirements.txt
└── README.md
```

## Quick start

Choose the path that works best for your setup:

| Path | GPU required | Docker required | Best for |
|------|--------------|-----------------|----------|
| **1. Google Colab** | No (free GPU provided) | No | Zero installation, quick start |
| **2. Dev container** | Recommended | Yes | Full development environment |
| **3. Native Python** | Optional | No | Direct local installation |
| **4. Conda** | Optional | No | Conda users |

### Path 1: Google Colab (easiest, no installation)

Perfect for students without local GPU or who want to start immediately.

1. **Open the Colab notebook**
   - Go to [Google Colab](https://colab.research.google.com/)
   - File → Open notebook → GitHub tab
   - Enter: `gperdrizet/GANNs-with-freinds`
   - Select: `notebooks/run_worker_colab.ipynb`

2. **Enable GPU runtime**
   - Runtime → Change runtime type
   - Hardware accelerator → GPU → T4
   - Save

3. **Run all cells**
   - Runtime → Run all
   - Follow prompts to configure database credentials
   - Training starts automatically

Your Colab GPU is now part of the training cluster!

### Path 2: Dev container (recommended for development)

Full development environment with GPU support.

**Prerequisites:**
- Docker with GPU support installed
- NVIDIA drivers (version ≥545)
- VS Code with Dev Containers extension

**Steps:**

1. **Fork and clone this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
   cd GANNs-with-freinds
   ```

2. **Open in dev container**
   - Open VS Code
   - Click 'Reopen in Container' when prompted
   - Wait for container to build

3. **Download CelebA dataset**
   ```bash
   python scripts/download_celeba.py
   ```
   This will automatically download the dataset using torchvision (~1.4 GB)

4. **Configure database connection**
   ```bash
   cp config.yaml.template config.yaml
   ```
   Edit `config.yaml` with database credentials provided by instructor

5. **Start worker**
   ```bash
   python src/worker.py
   ```

### Path 3: Native Python (no Docker needed)

Direct installation on your system.

**Prerequisites:**
- Python 3.10 or later
- pip package manager
- NVIDIA GPU + CUDA 11.8 (or CPU-only, see below)

**Steps:**

1. **Fork and clone this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
   cd GANNs-with-freinds
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   
   With GPU (CUDA 11.8):
   ```bash
   pip install -r requirements.txt
   ```
   
   CPU-only (no GPU):
   ```bash
   pip install -r requirements-cpu.txt
   ```

4. **Download CelebA dataset**
   ```bash
   python scripts/download_celeba.py
   ```

5. **Configure database connection**
   ```bash
   cp config.yaml.template config.yaml
   ```
   Edit `config.yaml` with database credentials provided by instructor

6. **Start worker**
   ```bash
   python src/worker.py
   ```

### Path 4: Conda environment

For conda users.

**Prerequisites:**
- Anaconda or Miniconda installed
- NVIDIA GPU + drivers (or CPU-only, see below)

**Steps:**

1. **Fork and clone this repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/GANNs-with-freinds.git
   cd GANNs-with-freinds
   ```

2. **Create conda environment**
   
   With GPU:
   ```bash
   conda env create -f environment.yml
   conda activate ganns-with-friends
   ```
   
   CPU-only (no GPU):
   ```bash
   conda env create -f environment-cpu.yml
   conda activate ganns-with-friends-cpu
   ```

3. **Download CelebA dataset**
   ```bash
   python scripts/download_celeba.py
   ```

4. **Configure database connection**
   ```bash
   cp config.yaml.template config.yaml
   ```
   Edit `config.yaml` with database credentials provided by instructor

5. **Start worker**
   ```bash
   python src/worker.py
   ```

### Alternative: Local training (single GPU, no database)

Want to train the same model locally without the distributed setup? Great for experimentation and comparison!

1. **Follow any setup path above** (skip database configuration)

2. **Start local training**
   ```bash
   python src/train_local.py --epochs 50 --batch-size 128
   ```

   Arguments:
   - `--epochs`: Number of epochs (default: 50)
   - `--batch-size`: Batch size (default: 128)
   - `--dataset-path`: Path to dataset (default: ../data/celeba)
   - `--output-dir`: Output directory (default: outputs_local)
   - `--sample-interval`: Generate samples every N epochs (default: 1)
   - `--checkpoint-interval`: Save checkpoint every N epochs (default: 5)
   - `--resume`: Resume from checkpoint path

3. **Monitor progress**
   - Generated samples: `outputs_local/samples/`
   - Checkpoints: `outputs_local/checkpoints/`

4. **View results**
   ```bash
   jupyter notebook notebooks/demo_trained_model.ipynb
   ```

### For instructor (main coordinator)

1. **Setup PostgreSQL database**
   - Deploy public facing SQL database
   - Create credentials for each student
   - Create shared table

2. **Follow any student setup path above**

3. **Initialize database**
   ```bash
   python src/database/init_db.py
   ```

4. **Start main coordinator**
   ```bash
   python src/main.py --epochs 50 --sample-interval 1
   ```

   Arguments:
   - `--epochs`: Number of epochs to train (default: 50)
   - `--sample-interval`: Generate samples every N iterations (default: 1)
   - `--config`: Path to config file (default: config.yaml)

5. **Monitor progress**
   - Generated samples appear in `outputs/samples/`
   - Check database for worker statistics
   - Training stops automatically when complete

6. **View results**
   ```bash
   jupyter notebook notebooks/demo_trained_model.ipynb
   ```

## Requirements

Requirements vary by installation path:

### Minimum (all paths)
- **10GB free disk space** (for CelebA dataset)
- **Internet connection** (for dataset download and database access)

### Hardware (optional)
- **NVIDIA GPU** (recommended for faster training, any consumer GPU works)
- **4GB+ VRAM** (for GPU training with default batch sizes)
- **CPU-only** (works but slower, automatic detection in worker)

### Software (path-dependent)

**Path 1 (Google Colab):**
- Google account only, everything else provided

**Path 2 (Dev container):**
- Docker with GPU support
- VS Code with Dev Containers extension
- NVIDIA drivers (version ≥545) for GPU support

**Path 3 (Native Python):**
- Python 3.10 or later
- pip package manager
- NVIDIA drivers for GPU (or CPU-only mode)

**Path 4 (Conda):**
- Anaconda or Miniconda
- NVIDIA drivers for GPU (or CPU-only mode)

## Configuration

Edit `config.yaml`:

```yaml
database:
  host: YOUR_DATABASE_HOST  # Provided by instructor
  port: 5432
  database: distributed_gan
  user: YOUR_DATABASE_USER  # Provided by instructor
  password: YOUR_DATABASE_PASSWORD  # Provided by instructor

training:
  batch_size: 32
  batches_per_work_unit: 10
  num_workers_per_update: 3

worker:
  poll_interval: 5  # seconds
  heartbeat_interval: 30  # seconds
  work_unit_timeout: 300  # seconds

data:
  dataset_path: ./data/celeba
```

## Hugging Face Hub integration (optional)

Push model checkpoints to Hugging Face Hub during training so students can view progress in real-time!

### Setup (for instructors)

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)

2. **Create a new model repository**
   - Go to [huggingface.co/new](https://huggingface.co/new)
   - Repository name: `distributed-gan-celeba`
   - Make it public so students can access it

3. **Get your access token**
   - Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   - Create a new token with write permissions

4. **Add to config.yaml**
   ```yaml
   huggingface:
     enabled: true
     repo_id: YOUR_USERNAME/distributed-gan-celeba
     token: YOUR_HF_TOKEN
     push_interval: 5  # Push every 5 iterations
   ```

5. **Install huggingface-hub**
   ```bash
   pip install huggingface-hub
   ```

### For students

Students can view live training progress by opening the demo notebook and setting:

```python
USE_HUGGINGFACE = True
HF_REPO_ID = 'instructor-username/distributed-gan-celeba'
```

The notebook will automatically download the latest checkpoint and display:
- Current training iteration and epoch
- Generated face samples
- Training loss curves (when available)

**Benefits:**
- See model improving in real-time during training
- No need to wait for training to complete
- Students can explore results even if they weren't able to contribute as workers
- Automatically handles checkpoint downloads

## What students learn

### Distributed systems
- Coordination without direct networking
- Fault tolerance and worker dropout
- Atomic operations and race conditions
- Work unit timeouts and reassignment

### Deep learning
- GAN architecture (generator and discriminator)
- Data parallel training
- Gradient aggregation
- Optimizer state management

### Database as message queue
- Novel use of PostgreSQL for distributed computing
- BLOB storage for model weights/gradients
- Atomic work unit claiming with `FOR UPDATE SKIP LOCKED`

## Monitoring

### For students
Check your contribution stats in the database or wait for the instructor's dashboard.

### For instructor
Monitor training progress via SQL queries:

```sql
-- Check training state
SELECT * FROM training_state;

-- See active workers
SELECT worker_id, gpu_name, total_images, last_heartbeat
FROM workers
WHERE last_heartbeat > NOW() - INTERVAL '2 minutes';

-- Work unit progress
SELECT status, COUNT(*) 
FROM work_units 
WHERE iteration = (SELECT current_iteration FROM training_state)
GROUP BY status;
```

## Troubleshooting

**Worker can't connect to database:**
- Verify `config.yaml` has correct credentials
- Check database is publicly accessible
- Test connection: `psql -h HOST -U USER -d DATABASE`

**Worker runs out of memory:**
- Reduce `batch_size` in `config.yaml`
- Reduce `num_workers_dataloader` to 2 or 0
- Close other GPU applications

**No work units available:**
- Training may not have started yet
- All work units may be claimed by other workers
- Check if training is still active in database

**Gradients not being aggregated:**
- Check that minimum number of workers have completed work units
- Verify `num_workers_per_update` setting
- Look for errors in main coordinator logs

## Testing

Test individual components:

```bash
# Test dataset loader
cd src/data
python dataset.py ../../data/celeba

# Test models
cd src/models
python dcgan.py

# Test database connection
cd src/database
python init_db.py

# Test utilities
cd src
python utils.py

# Test local training (quick 1-epoch test)
cd src
python train_local.py --epochs 1
```

## Distributed vs local training

### When to use distributed (main + workers)
- **Educational focus** - Learn distributed systems concepts  
- **Collaborative project** - Entire class working together  
- **Demonstrate real-world** - How large-scale training works  
- **Limited individual resources** - Pool multiple GPUs  

**Trade-offs:**
- Network overhead (database I/O)
- Coordination complexity
- Requires database setup

### When to use local training
- **Quick experimentation** - Test hyperparameters  
- **Solo learning** - Practice GANs independently  
- **Baseline comparison** - Compare distributed efficiency  
- **Faster iteration** - No network/coordination overhead  

**Trade-offs:**
- Limited to single GPU
- Misses distributed systems lessons

## Performance tips

### For students
- Close unnecessary applications
- Use dedicated GPU if you have multiple
- Reduce batch size if running low on VRAM
- Keep worker running continuously for best results

### For instructor  
- Start with more workers than `num_workers_per_update`
- Monitor for stale workers and timed-out work units
- Generate samples frequently to track progress
- Save checkpoints periodically

## Further reading

- [DCGAN Paper](https://arxiv.org/abs/1511.06434) - Original architecture
- [Data Parallel Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) - PyTorch guide
- [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) - Dataset details

## Contributing

This is an educational project! Contributions welcome:
- Bug fixes and improvements
- Additional GAN architectures
- Web-based monitoring dashboard
- Gradient compression techniques
- Support for other datasets

## License

MIT License - See LICENSE file for details

