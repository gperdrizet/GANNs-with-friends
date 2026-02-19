"""
Training Dashboard - Real-time monitoring for distributed GAN training.

A Streamlit dashboard that displays training progress, worker status,
and generated samples.

Usage:
    streamlit run src/dashboard.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_config, build_db_url
from database.db_manager import DatabaseManager


# Page configuration
st.set_page_config(
    page_title="GANNs with Friends - Dashboard",
    page_icon="G",
    layout="wide"
)

# Estimated FLOPs per image for DCGAN training (forward + backward)
# Generator: ~454 MFLOPs forward, Discriminator: ~214 MFLOPs forward (x2 for real/fake)
# Backward passes ~2x forward, total ~2.6 GFLOPs per image
GFLOPS_PER_IMAGE = 2.6


@st.cache_resource
def get_database():
    """Get database connection (cached)."""
    config = load_config('config.yaml')
    db_url = build_db_url(config['database'])
    return DatabaseManager(db_url)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_sample_images(samples_dir: Path, limit: int = 8):
    """Get the most recent sample images."""
    if not samples_dir.exists():
        return []
    
    images = sorted(samples_dir.glob('iteration_*.png'), reverse=True)
    return images[:limit]


def main():
    """Main dashboard."""
    
    st.title("GANNs with Friends")
    st.markdown("### Distributed GAN training dashboard")
    
    # Get database connection
    try:
        db = get_database()
    except Exception as e:
        st.error(f"Could not connect to database: {e}")
        st.info("Make sure config.yaml has valid database credentials.")
        return
    
    # Auto-refresh
    refresh_interval = st.sidebar.selectbox(
        "Auto-refresh",
        options=[0, 5, 10, 30, 60],
        format_func=lambda x: "Off" if x == 0 else f"Every {x}s",
        index=2
    )
    
    if refresh_interval > 0:
        st.sidebar.info(f"Refreshing every {refresh_interval} seconds")
        # Use st.empty for auto-refresh countdown
        import time
        
    # Get training state (handle empty database)
    training_state = db.get_training_state() or {}
    
    # ==================== Training Stats ====================
    st.markdown("---")
    st.subheader("Training status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if not training_state:
            status = "Not started"
        elif training_state.get('training_active'):
            status = "Active"
        else:
            status = "Stopped"
        st.metric("Status", status)
    
    with col2:
        st.metric("Iteration", training_state.get('current_iteration') or 0)
    
    with col3:
        st.metric("Epoch", (training_state.get('current_epoch') or 0) + 1)
    
    with col4:
        total_images = training_state.get('total_images_processed') or 0
        st.metric("Images processed", f"{total_images:,}")
    
    with col5:
        # Count active workers
        active_workers = db.get_active_workers(timeout_seconds=120)
        st.metric("Active workers", len(active_workers))
    
    # Current losses
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        g_loss = training_state.get('g_loss')
        st.metric("Generator loss", f"{g_loss:.4f}" if g_loss else "—")
    with col2:
        d_loss = training_state.get('d_loss')
        st.metric("Discriminator loss", f"{d_loss:.4f}" if d_loss else "—")
    with col3:
        d_real = training_state.get('d_real_acc')
        st.metric("D real accuracy", f"{d_real:.1%}" if d_real else "—")
    with col4:
        d_fake = training_state.get('d_fake_acc')
        st.metric("D fake accuracy", f"{d_fake:.1%}" if d_fake else "—")
    
    # ==================== Learning Curves ====================
    st.markdown("---")
    st.subheader("Learning curves")
    
    loss_history = db.get_loss_history()
    
    if loss_history:
        df = pd.DataFrame(loss_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Generator & discriminator loss**")
            loss_df = df[['iteration', 'g_loss', 'd_loss']].melt(
                id_vars=['iteration'],
                var_name='Model',
                value_name='Loss'
            )
            loss_df['Model'] = loss_df['Model'].map({
                'g_loss': 'Generator',
                'd_loss': 'Discriminator'
            })
            st.line_chart(
                loss_df.pivot(index='iteration', columns='Model', values='Loss'),
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Discriminator accuracy**")
            if 'd_real_acc' in df.columns and df['d_real_acc'].notna().any():
                acc_df = df[['iteration', 'd_real_acc', 'd_fake_acc']].melt(
                    id_vars=['iteration'],
                    var_name='Type',
                    value_name='Accuracy'
                )
                acc_df['Type'] = acc_df['Type'].map({
                    'd_real_acc': 'Real images',
                    'd_fake_acc': 'Fake images'
                })
                st.line_chart(
                    acc_df.pivot(index='iteration', columns='Type', values='Accuracy'),
                    use_container_width=True
                )
            else:
                st.info("Accuracy data not yet available")
    else:
        st.info("No training data yet. Start training to see learning curves.")
    
    # ==================== Sample Images ====================
    st.markdown("---")
    st.subheader("Generated samples")
    
    samples_dir = Path('data/outputs/samples')
    sample_images = get_sample_images(samples_dir, limit=8)
    
    if sample_images:
        # Show most recent samples
        cols = st.columns(4)
        for idx, img_path in enumerate(sample_images[:4]):
            with cols[idx]:
                iteration = img_path.stem.split('_')[-1]
                st.image(str(img_path), caption=f"Iteration {int(iteration)}", use_container_width=True)
        
        # Expandable section for more samples
        if len(sample_images) > 4:
            with st.expander("Show more samples"):
                cols = st.columns(4)
                for idx, img_path in enumerate(sample_images[4:8]):
                    with cols[idx]:
                        iteration = img_path.stem.split('_')[-1]
                        st.image(str(img_path), caption=f"Iteration {int(iteration)}", use_container_width=True)
    else:
        st.info("No sample images yet. Samples are generated during training.")
    
    # ==================== Worker Leaderboard ====================
    st.markdown("---")
    st.subheader("Worker leaderboard")
    
    all_workers = db.get_all_workers()
    
    if all_workers:
        # Create dataframe for display
        worker_df = pd.DataFrame(all_workers)
        
        # Calculate time since last heartbeat
        now = datetime.utcnow()
        worker_df['last_seen'] = worker_df['last_heartbeat'].apply(
            lambda x: format_duration((now - x).total_seconds()) + " ago" if x else "Never"
        )
        
        # Status indicator
        def status_indicator(row):
            if row['last_heartbeat']:
                seconds_ago = (now - row['last_heartbeat']).total_seconds()
                if seconds_ago < 60:
                    return "Active"
                elif seconds_ago < 300:
                    return "Idle"
            return "Offline"
        
        worker_df['status_icon'] = worker_df.apply(status_indicator, axis=1)
        
        # Format for display
        display_df = worker_df[[
            'status_icon', 'worker_id', 'hostname', 'gpu_name', 
            'total_work_units', 'total_images', 'last_seen'
        ]].copy()
        
        display_df.columns = ['Status', 'Worker ID', 'Name', 'GPU', 'Work units', 'Images', 'Last seen']
        
        # Truncate worker ID for display
        display_df['Worker ID'] = display_df['Worker ID'].str[:12] + '...'
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Summary stats
        total_work_units = worker_df['total_work_units'].sum()
        total_images = worker_df['total_images'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total workers (all time)", len(all_workers))
        with col2:
            st.metric("Total work units completed", f"{total_work_units:,}")
        with col3:
            st.metric("Total images processed", f"{total_images:,}")
        
        # ==================== Cluster Resources ====================
        st.markdown("---")
        st.subheader("Cluster resources")
        
        # Aggregate system resources from all workers
        total_cpu_cores = sum(w.get('cpu_cores') or 0 for w in all_workers)
        total_ram_gb = sum(w.get('ram_gb') or 0 for w in all_workers)
        total_gpu_vram_gb = sum(w.get('gpu_vram_gb') or 0 for w in all_workers)
        gpu_count = sum(1 for w in all_workers if w.get('gpu_vram_gb'))
        
        # Check for incomplete data
        workers_missing_cpu = sum(1 for w in all_workers if w.get('cpu_cores') is None)
        workers_missing_ram = sum(1 for w in all_workers if w.get('ram_gb') is None)
        has_incomplete_data = workers_missing_cpu > 0 or workers_missing_ram > 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            label = "Total CPU cores" + ("*" if workers_missing_cpu else "")
            st.metric(label, f"{total_cpu_cores:,}")
        with col2:
            label = "Total RAM" + ("*" if workers_missing_ram else "")
            st.metric(label, f"{total_ram_gb:,.1f} GB")
        with col3:
            st.metric("GPUs", f"{gpu_count}")
        with col4:
            st.metric("Total GPU VRAM", f"{total_gpu_vram_gb:,.1f} GB")
        
        if has_incomplete_data:
            missing_parts = []
            if workers_missing_cpu:
                missing_parts.append(f"{workers_missing_cpu} worker(s) missing CPU data")
            if workers_missing_ram:
                missing_parts.append(f"{workers_missing_ram} worker(s) missing RAM data")
            st.caption(f"*Incomplete: {', '.join(missing_parts)}. Workers may need to update to report system info.")
        
        # ==================== Throughput ====================
        st.markdown("---")
        st.subheader("Throughput")
        
        # Calculate images per second from recent history
        loss_history = db.get_loss_history()
        
        if len(loss_history) >= 2:
            # Get timestamps from training state updates
            recent_workers = [w for w in all_workers if w.get('last_heartbeat')]
            if recent_workers:
                # Estimate throughput from total images and training duration
                first_worker = min(recent_workers, key=lambda w: w['created_at'])
                training_duration = (datetime.utcnow() - first_worker['created_at']).total_seconds()
                
                if training_duration > 0:
                    images_per_sec = total_images / training_duration
                    gflops = images_per_sec * GFLOPS_PER_IMAGE
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Images/second", f"{images_per_sec:.1f}")
                    with col2:
                        if gflops >= 1000:
                            st.metric("Compute throughput", f"{gflops/1000:.2f} TFLOPS")
                        else:
                            st.metric("Compute throughput", f"{gflops:.1f} GFLOPS")
                    with col3:
                        st.metric("Training time", format_duration(training_duration))
                else:
                    st.info("Calculating throughput...")
            else:
                st.info("Waiting for worker data...")
        else:
            st.info("Need more training data to calculate throughput.")
            
    else:
        st.info("No workers have connected yet.")
    
    # ==================== Footer ====================
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "GANNs with Friends - Educational distributed GAN training | "
        "<a href='https://github.com/gperdrizet/GANNs-with-friends'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Auto-refresh using experimental_rerun
    if refresh_interval > 0:
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
