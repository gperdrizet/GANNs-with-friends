# Google Colab setup

Run the project on Google Colab with zero local installation required.

## Advantages

- No local installation needed
- Free GPU access (Tesla T4)
- Works on any device with a web browser
- Automatic dependency management
- Perfect for quick start

## Limitations

- Session timeout after inactivity
- Need to re-run setup after disconnect
- Limited to Colab's GPU quota
- Requires Google account

## Setup steps

### 1. Open the Colab notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File > Open notebook**
3. Select the **GitHub** tab
4. Enter: `gperdrizet/GANNs-with-freinds`
5. Select: `notebooks/run_worker_colab.ipynb`

### 2. Enable GPU runtime

1. Click **Runtime > Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Select **T4** (or best available)
4. Click **Save**

### 3. Run the setup cells

Execute the cells in order:

**Cell 1: Clone repository**
```python
# Clone repository if not already present
import os
if not os.path.exists('GANNs-with-freinds'):
    !git clone https://github.com/gperdrizet/GANNs-with-freinds.git
    %cd GANNs-with-freinds
else:
    %cd GANNs-with-freinds
    !git pull
```

**Cell 2: Install dependencies**
```python
!pip install -q -r requirements.txt
```

**Cell 3: Download dataset**
```python
!python scripts/download_celeba.py
```

This downloads ~1.3 GB of celebrity face images.

### 4. Configure database connection

**Cell 4: Create config file**
```python
if not os.path.exists('config.yaml'):
    !cp config.yaml.template config.yaml
    print('Created config.yaml from template')
    print('Edit config.yaml with your database credentials before continuing')
else:
    print('config.yaml already exists')
```

**Cell 5: Edit credentials**

Click the folder icon in the left sidebar, find `config.yaml`, and edit:

```yaml
database:
  host: YOUR_DATABASE_HOST      # From instructor
  port: 5432
  database: distributed_gan
  user: YOUR_USERNAME           # From instructor
  password: YOUR_PASSWORD       # From instructor
```

### 5. Start worker

**Cell 6: Run worker**
```python
!python src/worker.py
```

You should see:
```
Initializing worker...
Loading dataset...
Initializing models...
Worker abc123 initialized successfully!
GPU: Tesla T4
Dataset size: 202,599 images
Polling for work...
```

## Keeping the worker running

Colab sessions timeout after inactivity. To maximize uptime:

1. **Keep the browser tab active** - Don't close or switch away for long
2. **Monitor periodically** - Check every 30-60 minutes
3. **Use Colab Pro** (optional) - Longer runtimes and better GPUs
4. **Re-run when disconnected** - Just execute the cells again

## Monitoring your contribution

The worker prints updates as it processes batches:

```
Processing work unit 42 (iteration 5)...
Completed work unit 42 in 12.3s
Processed 320 images total
```

## Stopping the worker

To stop gracefully:
1. Click the **Stop** button in Colab
2. Or press **Runtime > Interrupt execution**

## Tips for Colab users

**Save your config:**
- Download `config.yaml` after editing
- On next session, upload it instead of editing again

**Monitor GPU usage:**
```python
!nvidia-smi
```

**Check remaining quota:**
- Colab shows GPU usage at bottom-right
- Free tier: ~12 hours/day
- Colab Pro: longer sessions

**Resume after disconnect:**
- Just re-run all cells
- Worker will pick up where training left off
- No data loss (all training state is in database)

## Troubleshooting

**GPU not available:**
- Verify runtime type is set to GPU
- May need to wait if quota exceeded
- Try again in a few hours

**Dataset download fails:**
- Re-run the download cell
- Check internet connection
- Try clearing output and re-running

**Can't connect to database:**
- Verify credentials in `config.yaml`
- Check database host is publicly accessible
- Contact instructor for help

**Session keeps disconnecting:**
- Normal for free tier with long idle periods
- Keep browser tab active
- Consider Colab Pro for longer sessions

## Next steps

- [Student guide](../guides/students.md) - Understanding your role as a worker
- [Monitoring](../guides/monitoring.md) - Track training progress
- [View results](../features/huggingface-integration.md) - See generated faces
