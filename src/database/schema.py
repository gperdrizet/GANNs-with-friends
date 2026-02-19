"""
Database schema for distributed GAN training.
Defines SQLAlchemy models for all tables.
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, 
    DateTime, LargeBinary, JSON, ForeignKey, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class ModelWeights(Base):
    """Stores current model weights for generator and discriminator."""
    __tablename__ = 'model_weights'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False)  # 'generator' or 'discriminator'
    iteration = Column(Integer, nullable=False, index=True)
    weights_blob = Column(LargeBinary, nullable=False)  # Serialized state_dict
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<ModelWeights(type={self.model_type}, iter={self.iteration})>'


class OptimizerState(Base):
    """Stores optimizer state for generator and discriminator."""
    __tablename__ = 'optimizer_state'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False)  # 'generator' or 'discriminator'
    iteration = Column(Integer, nullable=False, index=True)
    state_blob = Column(LargeBinary, nullable=False)  # Serialized optimizer state_dict
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<OptimizerState(type={self.model_type}, iter={self.iteration})>'


class Gradients(Base):
    """Stores gradients uploaded by workers."""
    __tablename__ = 'gradients'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String(100), ForeignKey('workers.worker_id'), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # 'generator' or 'discriminator'
    iteration = Column(Integer, nullable=False, index=True)
    gradients_blob = Column(LargeBinary, nullable=False)  # Serialized gradient tensors
    work_unit_id = Column(Integer, ForeignKey('work_units.id'), nullable=False)
    num_samples = Column(Integer, nullable=False)  # Number of samples used to compute gradients
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<Gradients(worker={self.worker_id}, type={self.model_type}, iter={self.iteration})>'


class WorkUnit(Base):
    """Represents a batch of work to be processed by a worker."""
    __tablename__ = 'work_units'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    iteration = Column(Integer, nullable=False, index=True)
    image_indices = Column(JSON, nullable=False)  # Array of image indices
    status = Column(String(20), nullable=False, default='pending', index=True)  
    # Status: 'pending', 'claimed', 'completed', 'failed'
    worker_id = Column(String(100), ForeignKey('workers.worker_id'), nullable=True)
    claimed_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    timeout_at = Column(DateTime, nullable=True)
    num_batches = Column(Integer, nullable=False)  # Number of batches in this work unit
    
    def __repr__(self):
        return f'<WorkUnit(id={self.id}, status={self.status}, worker={self.worker_id})>'


class TrainingState(Base):
    """Singleton table storing current training state."""
    __tablename__ = 'training_state'
    
    id = Column(Integer, primary_key=True, default=1)
    current_iteration = Column(Integer, nullable=False, default=0)
    current_epoch = Column(Integer, nullable=False, default=0)
    g_loss = Column(Float, nullable=True)
    d_loss = Column(Float, nullable=True)
    d_real_acc = Column(Float, nullable=True)
    d_fake_acc = Column(Float, nullable=True)
    total_batches_processed = Column(Integer, nullable=False, default=0)
    total_images_processed = Column(Integer, nullable=False, default=0)
    training_active = Column(Boolean, nullable=False, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<TrainingState(iter={self.current_iteration}, epoch={self.current_epoch})>'


class LossHistory(Base):
    """Stores loss values at each iteration for learning curves."""
    __tablename__ = 'loss_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    iteration = Column(Integer, nullable=False, index=True, unique=True)
    epoch = Column(Integer, nullable=False)
    g_loss = Column(Float, nullable=False)
    d_loss = Column(Float, nullable=False)
    d_real_acc = Column(Float, nullable=True)
    d_fake_acc = Column(Float, nullable=True)
    num_workers = Column(Integer, nullable=True)  # Workers that contributed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<LossHistory(iter={self.iteration}, g_loss={self.g_loss:.4f}, d_loss={self.d_loss:.4f})>'


class Worker(Base):
    """Stores information about each worker."""
    __tablename__ = 'workers'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    worker_id = Column(String(100), unique=True, nullable=False, index=True)
    hostname = Column(String(255), nullable=True)
    gpu_name = Column(String(255), nullable=True)
    status = Column(String(20), nullable=False, default='active')  # 'active', 'idle', 'offline'
    total_work_units = Column(Integer, nullable=False, default=0)
    total_batches = Column(Integer, nullable=False, default=0)
    total_images = Column(Integer, nullable=False, default=0)
    last_heartbeat = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<Worker(id={self.worker_id}, status={self.status})>'
