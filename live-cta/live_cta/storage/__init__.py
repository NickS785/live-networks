"""
CTAFlow Storage System

Provides unified storage and retrieval for:
- Trained models
- Training data
- Predictions and backtests
- Configuration files

Supports:
- Local filesystem storage
- AWS S3 cloud storage
- Hybrid caching strategies
"""

from .model_manager import ModelManager, ModelMode
from .aws_client import AWSClient, S3Config
from .storage_backend import StorageBackend, LocalStorage, S3Storage

__all__ = [
    'ModelManager',
    'ModelMode',
    'AWSClient',
    'S3Config',
    'StorageBackend',
    'LocalStorage',
    'S3Storage',
]
