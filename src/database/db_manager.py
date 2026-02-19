"""
Database manager for distributed GAN training.
Handles all database operations and connections.
"""

import io
import pickle
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager

import torch
from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from .schema import (
    Base, ModelWeights, OptimizerState, Gradients, 
    WorkUnit, TrainingState, Worker, LossHistory
)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_url: str):
        """Initialize database connection.
        
        Args:
            db_url: PostgreSQL connection string
        """
        self.engine = create_engine(
            db_url,
            poolclass=NullPool,  # No connection pooling for simplicity
            pool_pre_ping=True,  # Verify connections before using
            connect_args={
                'options': '-c statement_timeout=0'  # Disable statement timeout
            }
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def init_database(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
    
    def reset_database(self):
        """Drop and recreate all tables. Use with caution!"""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
    
    # ==================== Model Weights ====================
    
    def save_model_weights(self, model_type: str, iteration: int, state_dict: Dict[str, torch.Tensor]):
        """Save model weights to database.
        
        Args:
            model_type: 'generator' or 'discriminator'
            iteration: Current training iteration
            state_dict: PyTorch model state_dict
        """
        weights_blob = pickle.dumps(state_dict)
        
        with self.get_session() as session:
            weights = ModelWeights(
                model_type=model_type,
                iteration=iteration,
                weights_blob=weights_blob
            )
            session.add(weights)
    
    def get_latest_model_weights(self, model_type: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get latest model weights from database.
        
        Args:
            model_type: 'generator' or 'discriminator'
            
        Returns:
            PyTorch state_dict or None if not found
        """
        with self.get_session() as session:
            weights = session.query(ModelWeights).filter(
                ModelWeights.model_type == model_type
            ).order_by(ModelWeights.iteration.desc()).first()
            
            if weights:
                return pickle.loads(weights.weights_blob)
            return None
    
    def get_model_weights_at_iteration(self, model_type: str, iteration: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get model weights at specific iteration.
        
        Args:
            model_type: 'generator' or 'discriminator'
            iteration: Training iteration
            
        Returns:
            PyTorch state_dict or None if not found
        """
        with self.get_session() as session:
            weights = session.query(ModelWeights).filter(
                and_(
                    ModelWeights.model_type == model_type,
                    ModelWeights.iteration == iteration
                )
            ).first()
            
            if weights:
                return pickle.loads(weights.weights_blob)
            return None
    
    # ==================== Optimizer State ====================
    
    def save_optimizer_state(self, model_type: str, iteration: int, state_dict: Dict):
        """Save optimizer state to database.
        
        Args:
            model_type: 'generator' or 'discriminator'
            iteration: Current training iteration
            state_dict: PyTorch optimizer state_dict
        """
        state_blob = pickle.dumps(state_dict)
        
        with self.get_session() as session:
            optimizer_state = OptimizerState(
                model_type=model_type,
                iteration=iteration,
                state_blob=state_blob
            )
            session.add(optimizer_state)
    
    def get_latest_optimizer_state(self, model_type: str) -> Optional[Dict]:
        """Get latest optimizer state from database.
        
        Args:
            model_type: 'generator' or 'discriminator'
            
        Returns:
            PyTorch optimizer state_dict or None if not found
        """
        with self.get_session() as session:
            state = session.query(OptimizerState).filter(
                OptimizerState.model_type == model_type
            ).order_by(OptimizerState.iteration.desc()).first()
            
            if state:
                return pickle.loads(state.state_blob)
            return None
    
    # ==================== Gradients ====================
    
    def save_gradients(
        self, 
        worker_id: str, 
        model_type: str, 
        iteration: int,
        work_unit_id: int,
        gradients: Dict[str, torch.Tensor],
        num_samples: int
    ):
        """Save gradients from a worker.
        
        Args:
            worker_id: Unique worker identifier
            model_type: 'generator' or 'discriminator'
            iteration: Current training iteration
            work_unit_id: ID of completed work unit
            gradients: Dictionary of parameter gradients
            num_samples: Number of samples used to compute gradients
        """
        gradients_blob = pickle.dumps(gradients)
        
        with self.get_session() as session:
            grad = Gradients(
                worker_id=worker_id,
                model_type=model_type,
                iteration=iteration,
                work_unit_id=work_unit_id,
                gradients_blob=gradients_blob,
                num_samples=num_samples
            )
            session.add(grad)
    
    def get_gradients_for_iteration(
        self, 
        model_type: str, 
        iteration: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get all gradients for a specific iteration.
        
        Args:
            model_type: 'generator' or 'discriminator'
            iteration: Training iteration
            limit: Maximum number of gradients to return
            
        Returns:
            List of dictionaries containing gradient info
        """
        with self.get_session() as session:
            query = session.query(Gradients).filter(
                and_(
                    Gradients.model_type == model_type,
                    Gradients.iteration == iteration
                )
            )
            
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            
            return [{
                'worker_id': g.worker_id,
                'gradients': pickle.loads(g.gradients_blob),
                'num_samples': g.num_samples,
                'work_unit_id': g.work_unit_id
            } for g in results]
    
    def delete_gradients_for_iteration(self, iteration: int):
        """Delete all gradients for a specific iteration (cleanup after aggregation)."""
        with self.get_session() as session:
            session.query(Gradients).filter(
                Gradients.iteration == iteration
            ).delete()
    
    # ==================== Work Units ====================
    
    def create_work_units(
        self, 
        iteration: int, 
        image_indices_list: List[List[int]],
        num_batches_per_unit: int,
        timeout_seconds: int = 300
    ) -> List[int]:
        """Create multiple work units.
        
        Args:
            iteration: Training iteration
            image_indices_list: List of lists of image indices
            num_batches_per_unit: Number of batches in each work unit
            timeout_seconds: Timeout in seconds
            
        Returns:
            List of created work unit IDs
        """
        work_unit_ids = []
        
        with self.get_session() as session:
            for image_indices in image_indices_list:
                timeout_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)
                work_unit = WorkUnit(
                    iteration=iteration,
                    image_indices=image_indices,
                    status='pending',
                    num_batches=num_batches_per_unit,
                    timeout_at=timeout_at
                )
                session.add(work_unit)
                session.flush()  # Get the ID
                work_unit_ids.append(work_unit.id)
        
        return work_unit_ids
    
    def claim_work_unit(self, worker_id: str, timeout_seconds: int = 300) -> Optional[Dict[str, Any]]:
        """Atomically claim an available work unit.
        
        Args:
            worker_id: Unique worker identifier
            timeout_seconds: Timeout in seconds
            
        Returns:
            Dictionary with work unit info or None if no work available
        """
        with self.get_session() as session:
            # Find pending or timed-out work units
            now = datetime.utcnow()
            work_unit = session.query(WorkUnit).filter(
                or_(
                    WorkUnit.status == 'pending',
                    and_(
                        WorkUnit.status == 'claimed',
                        WorkUnit.timeout_at < now
                    )
                )
            ).with_for_update(skip_locked=True).first()
            
            if work_unit:
                work_unit.status = 'claimed'
                work_unit.worker_id = worker_id
                work_unit.claimed_at = now
                work_unit.timeout_at = now + timedelta(seconds=timeout_seconds)
                
                return {
                    'id': work_unit.id,
                    'iteration': work_unit.iteration,
                    'image_indices': work_unit.image_indices,
                    'num_batches': work_unit.num_batches
                }
            
            return None
    
    def complete_work_unit(self, work_unit_id: int):
        """Mark a work unit as completed."""
        with self.get_session() as session:
            work_unit = session.query(WorkUnit).filter(
                WorkUnit.id == work_unit_id
            ).first()
            
            if work_unit:
                work_unit.status = 'completed'
                work_unit.completed_at = datetime.utcnow()
    
    def cancel_pending_work_units(self, iteration: int) -> int:
        """Cancel all pending work units for a given iteration.
        
        This is called when the coordinator aggregates gradients and moves
        to the next iteration, so workers don't waste time on stale work.
        
        Args:
            iteration: Training iteration to cancel work units for
            
        Returns:
            Number of work units cancelled
        """
        with self.get_session() as session:
            result = session.query(WorkUnit).filter(
                WorkUnit.iteration == iteration,
                WorkUnit.status.in_(['pending', 'claimed'])
            ).update(
                {'status': 'cancelled'},
                synchronize_session=False
            )
            return result
    
    def get_work_unit_stats(self, iteration: int) -> Dict[str, int]:
        """Get statistics about work units for an iteration.
        
        Returns:
            Dictionary with counts for each status
        """
        with self.get_session() as session:
            work_units = session.query(WorkUnit).filter(
                WorkUnit.iteration == iteration
            ).all()
            
            stats = {
                'pending': 0,
                'claimed': 0,
                'completed': 0,
                'failed': 0,
                'total': len(work_units)
            }
            
            for wu in work_units:
                stats[wu.status] = stats.get(wu.status, 0) + 1
            
            return stats
    
    # ==================== Training State ====================
    
    def init_training_state(self):
        """Initialize training state (singleton)."""
        with self.get_session() as session:
            state = session.query(TrainingState).filter(TrainingState.id == 1).first()
            if not state:
                state = TrainingState(id=1)
                session.add(state)
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        with self.get_session() as session:
            state = session.query(TrainingState).filter(TrainingState.id == 1).first()
            if state:
                return {
                    'current_iteration': state.current_iteration,
                    'current_epoch': state.current_epoch,
                    'g_loss': state.g_loss,
                    'd_loss': state.d_loss,
                    'd_real_acc': state.d_real_acc,
                    'd_fake_acc': state.d_fake_acc,
                    'total_batches_processed': state.total_batches_processed,
                    'total_images_processed': state.total_images_processed,
                    'training_active': state.training_active
                }
            return None
    
    def update_training_state(self, **kwargs):
        """Update training state fields."""
        with self.get_session() as session:
            state = session.query(TrainingState).filter(TrainingState.id == 1).first()
            if state:
                for key, value in kwargs.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
    
    # ==================== Workers ====================
    
    def register_worker(self, worker_id: str, hostname: str = None, gpu_name: str = None):
        """Register a new worker or update existing one."""
        with self.get_session() as session:
            worker = session.query(Worker).filter(Worker.worker_id == worker_id).first()
            
            if worker:
                # Update existing worker
                worker.status = 'active'
                worker.last_heartbeat = datetime.utcnow()
                if hostname:
                    worker.hostname = hostname
                if gpu_name:
                    worker.gpu_name = gpu_name
            else:
                # Create new worker
                worker = Worker(
                    worker_id=worker_id,
                    hostname=hostname,
                    gpu_name=gpu_name,
                    status='active'
                )
                session.add(worker)
    
    def update_worker_heartbeat(self, worker_id: str):
        """Update worker's last heartbeat timestamp."""
        with self.get_session() as session:
            worker = session.query(Worker).filter(Worker.worker_id == worker_id).first()
            if worker:
                worker.last_heartbeat = datetime.utcnow()
                worker.status = 'active'
    
    def update_worker_stats(self, worker_id: str, work_units: int = 0, batches: int = 0, images: int = 0):
        """Update worker statistics."""
        with self.get_session() as session:
            worker = session.query(Worker).filter(Worker.worker_id == worker_id).first()
            if worker:
                worker.total_work_units += work_units
                worker.total_batches += batches
                worker.total_images += images
    
    def get_active_workers(self, timeout_seconds: int = 120) -> List[Dict[str, Any]]:
        """Get list of active workers (heartbeat within timeout).
        
        Args:
            timeout_seconds: Consider worker offline after this many seconds
            
        Returns:
            List of dictionaries with worker info
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=timeout_seconds)
        
        with self.get_session() as session:
            workers = session.query(Worker).filter(
                Worker.last_heartbeat >= cutoff_time
            ).all()
            
            return [{
                'worker_id': w.worker_id,
                'hostname': w.hostname,
                'gpu_name': w.gpu_name,
                'status': w.status,
                'total_work_units': w.total_work_units,
                'total_batches': w.total_batches,
                'total_images': w.total_images,
                'last_heartbeat': w.last_heartbeat
            } for w in workers]
    
    def mark_stale_workers_offline(self, timeout_seconds: int = 120):
        """Mark workers as offline if they haven't sent heartbeat recently."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=timeout_seconds)
        
        with self.get_session() as session:
            session.query(Worker).filter(
                Worker.last_heartbeat < cutoff_time
            ).update({'status': 'offline'})
    
    # ==================== Loss History ====================
    
    def save_loss_history(
        self,
        iteration: int,
        epoch: int,
        g_loss: float,
        d_loss: float,
        d_real_acc: float = None,
        d_fake_acc: float = None,
        num_workers: int = None
    ):
        """Save loss values for an iteration.
        
        Args:
            iteration: Training iteration
            epoch: Current epoch
            g_loss: Generator loss
            d_loss: Discriminator loss
            d_real_acc: Discriminator accuracy on real images
            d_fake_acc: Discriminator accuracy on fake images
            num_workers: Number of workers that contributed gradients
        """
        with self.get_session() as session:
            # Update if exists, otherwise insert
            existing = session.query(LossHistory).filter(
                LossHistory.iteration == iteration
            ).first()
            
            if existing:
                existing.epoch = epoch
                existing.g_loss = g_loss
                existing.d_loss = d_loss
                existing.d_real_acc = d_real_acc
                existing.d_fake_acc = d_fake_acc
                existing.num_workers = num_workers
            else:
                entry = LossHistory(
                    iteration=iteration,
                    epoch=epoch,
                    g_loss=g_loss,
                    d_loss=d_loss,
                    d_real_acc=d_real_acc,
                    d_fake_acc=d_fake_acc,
                    num_workers=num_workers
                )
                session.add(entry)
    
    def get_loss_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get loss history for learning curves.
        
        Args:
            limit: Maximum number of entries to return (most recent)
            
        Returns:
            List of dictionaries with loss values, ordered by iteration
        """
        with self.get_session() as session:
            query = session.query(LossHistory).order_by(LossHistory.iteration)
            
            if limit:
                query = query.order_by(LossHistory.iteration.desc()).limit(limit)
                results = query.all()
                results.reverse()  # Back to ascending order
            else:
                results = query.all()
            
            return [{
                'iteration': h.iteration,
                'epoch': h.epoch,
                'g_loss': h.g_loss,
                'd_loss': h.d_loss,
                'd_real_acc': h.d_real_acc,
                'd_fake_acc': h.d_fake_acc,
                'num_workers': h.num_workers,
                'created_at': h.created_at
            } for h in results]
    
    def get_all_workers(self) -> List[Dict[str, Any]]:
        """Get all workers (for dashboard).
        
        Returns:
            List of dictionaries with worker info
        """
        with self.get_session() as session:
            workers = session.query(Worker).order_by(
                Worker.total_work_units.desc()
            ).all()
            
            return [{
                'worker_id': w.worker_id,
                'hostname': w.hostname,
                'gpu_name': w.gpu_name,
                'status': w.status,
                'total_work_units': w.total_work_units,
                'total_batches': w.total_batches,
                'total_images': w.total_images,
                'last_heartbeat': w.last_heartbeat,
                'created_at': w.created_at
            } for w in workers]
