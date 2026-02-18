# Monitoring

Guide to monitoring distributed GAN training progress.

## Overview

Monitoring helps you:
- Track training progress
- Identify issues early
- Optimize resource usage
- Verify worker participation
- Assess model quality

## Monitoring methods

### 1. Console output

**For workers:**
```
Processing work unit 42 (iteration 5)...
Completed work unit 42 in 15.3s
Processed 320 images total
```

**For coordinator:**
```
Waiting for workers... (2/3 completed)
Aggregating gradients from 3 workers...
Generator loss: 2.345
Discriminator loss: 1.234
Generated samples saved
```

### 2. Database queries

Real-time status from PostgreSQL.

### 3. Generated samples

Visual inspection of image quality.

### 4. Hugging Face

Live model updates (if enabled).

## Database monitoring

### Active workers

```sql
SELECT worker_id, gpu_name, total_work_units, 
       total_images, last_heartbeat
FROM workers
WHERE last_heartbeat > NOW() - INTERVAL '2 minutes'
ORDER BY total_work_units DESC;
```

Shows currently active workers.

### Training progress

```sql
SELECT current_iteration, current_epoch,
       generator_loss, discriminator_loss,
       samples_generated
FROM training_state;
```

Current status of training.

### Work unit status

```sql
SELECT status, COUNT(*) as count
FROM work_units
WHERE iteration = (SELECT current_iteration FROM training_state)
GROUP BY status;
```

Shows distribution: pending, in_progress, completed, failed.

### Worker contribution leaderboard

```sql
SELECT worker_id, 
       COUNT(*) as work_units,
       SUM(EXTRACT(EPOCH FROM (completed_at - claimed_at))) as total_seconds,
       MIN(completed_at) as first_contribution,
       MAX(completed_at) as last_contribution
FROM work_units
WHERE status = 'completed'
GROUP BY worker_id
ORDER BY work_units DESC
LIMIT 10;
```

Top contributors.

### Stalled work units

```sql
SELECT id, iteration, claimed_by, claimed_at,
       NOW() - claimed_at as age
FROM work_units
WHERE status = 'in_progress'
  AND claimed_at < NOW() - INTERVAL '10 minutes'
ORDER BY claimed_at;
```

Work units that may need reclaiming.

## Python monitoring script

Create `monitor.py`:

```python
#!/usr/bin/env python
import psycopg2
import time
from datetime import datetime
from src.utils import load_config, build_db_url

def get_stats(conn):
    """Get current training statistics."""
    with conn.cursor() as cur:
        # Training state
        cur.execute('SELECT current_iteration, current_epoch FROM training_state')
        iteration, epoch = cur.fetchone()
        
        # Active workers
        cur.execute('''
            SELECT COUNT(*) FROM workers
            WHERE last_heartbeat > NOW() - INTERVAL '2 minutes'
        ''')
        active_workers = cur.fetchone()[0]
        
        # Work units for current iteration
        cur.execute('''
            SELECT status, COUNT(*) FROM work_units
            WHERE iteration = %s
            GROUP BY status
        ''', (iteration,))
        work_status = dict(cur.fetchall())
        
        return {
            'iteration': iteration,
            'epoch': epoch,
            'active_workers': active_workers,
            'work_status': work_status
        }

def main():
    config = load_config('config.yaml')
    db_url = build_db_url(config['database'])
    conn = psycopg2.connect(db_url)
    
    try:
        while True:
            stats = get_stats(conn)
            
            print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Iteration: {stats['iteration']}, Epoch: {stats['epoch']}")
            print(f"Active workers: {stats['active_workers']}")
            
            ws = stats['work_status']
            total = sum(ws.values())
            completed = ws.get('completed', 0)
            in_progress = ws.get('in_progress', 0)
            pending = ws.get('pending', 0)
            
            print(f"Work units: {completed}/{total} completed, {in_progress} in progress, {pending} pending")
            
            if total > 0:
                progress = (completed / total) * 100
                print(f"Progress: {progress:.1f}%")
            
            time.sleep(10)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
    finally:
        conn.close()

if __name__ == '__main__':
    main()
```

Run with: `python monitor.py`

## Visual monitoring

### Watch generated samples

```bash
# View latest samples
ls -lt outputs/samples/ | head

# Display in terminal (if using imgcat or similar)
imgcat outputs/samples/iteration_0010.png
```

### Create GIF of progress

```python
from PIL import Image
import glob

# Load all sample images
images = []
for filepath in sorted(glob.glob('outputs/samples/iteration_*.png')):
    images.append(Image.open(filepath))

# Save as GIF
images[0].save(
    'training_progress.gif',
    save_all=True,
    append_images=images[1:],
    duration=500,
    loop=0
)
```

### Plot loss curves

```python
import psycopg2
import matplotlib.pyplot as plt

# Fetch loss history
conn = psycopg2.connect(db_url)
cur = conn.cursor()
cur.execute('''
    SELECT iteration, generator_loss, discriminator_loss
    FROM loss_history
    ORDER BY iteration
''')
data = cur.fetchall()

iterations, g_loss, d_loss = zip(*data)

plt.figure(figsize=(10, 5))
plt.plot(iterations, g_loss, label='Generator')
plt.plot(iterations, d_loss, label='Discriminator')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')
plt.savefig('loss_curves.png')
```

## Hugging Face monitoring

If enabled, view progress at:
```
https://huggingface.co/YOUR_USERNAME/YOUR_REPO
```

Download latest checkpoint:
```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id='instructor/distributed-gan',
    filename='checkpoint_latest.pth'
)
```

## Performance metrics

### Worker efficiency

```sql
SELECT worker_id,
       COUNT(*) as work_units,
       AVG(EXTRACT(EPOCH FROM (completed_at - claimed_at))) as avg_time,
       MIN(EXTRACT(EPOCH FROM (completed_at - claimed_at))) as min_time,
       MAX(EXTRACT(EPOCH FROM (completed_at - claimed_at))) as max_time
FROM work_units
WHERE status = 'completed'
GROUP BY worker_id;
```

### Database performance

```sql
-- Table sizes
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Active connections
SELECT COUNT(*) FROM pg_stat_activity;
```

## Alerts and notifications

### Email on completion

```python
import smtplib
from email.message import EmailMessage

def send_alert(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = 'training@example.com'
    msg['To'] = 'instructor@example.com'
    
    with smtplib.SMTP('localhost') as s:
        s.send_message(msg)

# In training loop
if current_epoch == target_epochs:
    send_alert(
        'Training Complete',
        f'GAN training finished after {total_iterations} iterations'
    )
```

### Slack notifications

```python
import requests

def slack_notify(message):
    webhook_url = 'YOUR_SLACK_WEBHOOK'
    requests.post(webhook_url, json={'text': message})

# On error
slack_notify(f'Worker {worker_id} failed: {error_msg}')
```

## Troubleshooting with monitoring

### Issue: No workers active

**Check:**
```sql
SELECT * FROM workers ORDER BY last_heartbeat DESC LIMIT 5;
```

**Solution:** Workers haven't started or all disconnected.

### Issue: Work units stuck

**Check:**
```sql
SELECT COUNT(*), status FROM work_units 
WHERE iteration = CURRENT_ITERATION
GROUP BY status;
```

**Solution:** Reclaim stalled units or check worker errors.

### Issue: Slow progress

**Check:**
```sql
SELECT AVG(completed_at - claimed_at) FROM work_units 
WHERE status = 'completed';
```

**Solution:** Workers may be overloaded or network is slow.

## Dashboard ideas

### Simple web dashboard

```python
from flask import Flask, render_template
import psycopg2

app = Flask(__name__)

@app.route('/')
def dashboard():
    conn = psycopg2.connect(db_url)
    stats = get_stats(conn)
    conn.close()
    return render_template('dashboard.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=True)
```

### Real-time updates with WebSockets

Use libraries like Flask-SocketIO for live updates.

## Best practices

- Monitor regularly during active training
- Set up alerts for critical issues
- Keep database queries efficient
- Archive old data periodically
- Share progress with students

## Next steps

- [Troubleshooting](../resources/troubleshooting.md) - Fix common issues
- [Performance](../resources/performance.md) - Optimize training
- [Architecture](../architecture/database.md) - Understand the database schema
