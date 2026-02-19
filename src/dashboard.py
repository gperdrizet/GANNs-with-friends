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
    page_icon="🎨",
    layout="wide"
)


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
    
    st.title("🎨 GANNs with Friends")
    st.markdown("### Distributed GAN Training Dashboard")
    
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
        
    # Get training state
    training_state = db.get_training_state()
    
    # ==================== Training Stats ====================
    st.markdown("---")
    st.subheader("📊 Training Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        status = "🟢 Active" if training_state.get('training_active') else "🔴 Stopped"
        st.metric("Status", status)
    
    with col2:
        st.metric("Iteration", training_state.get('current_iteration', 0))
    
    with col3:
        st.metric("Epoch", training_state.get('current_epoch', 0) + 1)
    
    with col4:
        total_images = training_state.get('total_images_processed', 0)
        st.metric("Images Processed", f"{total_images:,}")
    
    with col5:
        # Count active workers
        active_workers = db.get_active_workers(timeout_seconds=120)
        st.metric("Active Workers", len(active_workers))
    
    # Current losses
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        g_loss = training_state.get('g_loss')
        st.metric("Generator Loss", f"{g_loss:.4f}" if g_loss else "—")
    with col2:
        d_loss = training_state.get('d_loss')
        st.metric("Discriminator Loss", f"{d_loss:.4f}" if d_loss else "—")
    with col3:
        d_real = training_state.get('d_real_acc')
        st.metric("D Real Accuracy", f"{d_real:.1%}" if d_real else "—")
    with col4:
        d_fake = training_state.get('d_fake_acc')
        st.metric("D Fake Accuracy", f"{d_fake:.1%}" if d_fake else "—")
    
    # ==================== Learning Curves ====================
    st.markdown("---")
    st.subheader("📈 Learning Curves")
    
    loss_history = db.get_loss_history()
    
    if loss_history:
        df = pd.DataFrame(loss_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Generator & Discriminator Loss**")
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
            st.markdown("**Discriminator Accuracy**")
            if 'd_real_acc' in df.columns and df['d_real_acc'].notna().any():
                acc_df = df[['iteration', 'd_real_acc', 'd_fake_acc']].melt(
                    id_vars=['iteration'],
                    var_name='Type',
                    value_name='Accuracy'
                )
                acc_df['Type'] = acc_df['Type'].map({
                    'd_real_acc': 'Real Images',
                    'd_fake_acc': 'Fake Images'
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
    st.subheader("🖼️ Generated Samples")
    
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
    st.subheader("👥 Worker Leaderboard")
    
    all_workers = db.get_all_workers()
    
    if all_workers:
        # Create dataframe for display
        worker_df = pd.DataFrame(all_workers)
        
        # Calculate time since last heartbeat
        now = datetime.utcnow()
        worker_df['last_seen'] = worker_df['last_heartbeat'].apply(
            lambda x: format_duration((now - x).total_seconds()) + " ago" if x else "Never"
        )
        
        # Status emoji
        def status_emoji(row):
            if row['last_heartbeat']:
                seconds_ago = (now - row['last_heartbeat']).total_seconds()
                if seconds_ago < 60:
                    return "🟢"  # Active
                elif seconds_ago < 300:
                    return "🟡"  # Idle
            return "🔴"  # Offline
        
        worker_df['status_icon'] = worker_df.apply(status_emoji, axis=1)
        
        # Format for display
        display_df = worker_df[[
            'status_icon', 'worker_id', 'hostname', 'gpu_name', 
            'total_work_units', 'total_images', 'last_seen'
        ]].copy()
        
        display_df.columns = ['', 'Worker ID', 'Hostname', 'GPU', 'Work Units', 'Images', 'Last Seen']
        
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
            st.metric("Total Workers (all time)", len(all_workers))
        with col2:
            st.metric("Total Work Units Completed", f"{total_work_units:,}")
        with col3:
            st.metric("Total Images Processed", f"{total_images:,}")
    else:
        st.info("No workers have connected yet.")
    
    # ==================== Footer ====================
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "GANNs with Friends - Educational Distributed GAN Training | "
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
