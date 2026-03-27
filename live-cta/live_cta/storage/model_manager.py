"""
Model Manager for CTAFlow

Centralized management of model lifecycle:
- Training mode: Save checkpoints, track metrics
- Backtest mode: Load trained models, run simulations
- Production mode: Load production models, make predictions

Supports:
- Local and S3 storage
- Model versioning
- Checkpoint management
- Metadata tracking
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
from enum import Enum
from datetime import datetime
import json
import logging

from .storage_backend import StorageBackend, LocalStorage, S3Storage
from .aws_client import AWSClient, S3Config

logger = logging.getLogger(__name__)


class ModelMode(Enum):
    """Operating modes for ModelManager."""
    TRAINING = "training"
    BACKTEST = "backtest"
    PRODUCTION = "production"


class ModelManager:
    """
    Centralized model management for CTAFlow.

    Features:
    - Load/save models with metadata
    - Version control and checkpointing
    - Multi-backend storage (local, S3)
    - Production/backtest/training modes

    Parameters
    ----------
    storage_backend : StorageBackend
        Storage backend (LocalStorage or S3Storage)
    mode : ModelMode
        Operating mode
    models_dir : str
        Directory/prefix for model storage (default: 'models/')

    Example
    -------
    >>> # Training mode with local storage
    >>> from CTAFlow.data.storage import LocalStorage
    >>> storage = LocalStorage('data/')
    >>> manager = ModelManager(storage, mode=ModelMode.TRAINING)
    >>>
    >>> # Save a trained model
    >>> manager.save_model(
    ...     model=my_model,
    ...     model_name='wspr_HE_LE',
    ...     metadata={'tickers': ['HE', 'LE'], 'accuracy': 0.65}
    ... )
    >>>
    >>> # Load for inference
    >>> model, metadata = manager.load_model('wspr_HE_LE')
    >>>
    >>> # Production mode with S3
    >>> from CTAFlow.data.storage import AWSClient, S3Storage, S3Config
    >>> aws_client = AWSClient(S3Config(bucket_name='ctaflow-prod'))
    >>> storage = S3Storage('ctaflow-prod', aws_client)
    >>> manager = ModelManager(storage, mode=ModelMode.PRODUCTION)
    """

    def __init__(
        self,
        storage_backend: StorageBackend,
        mode: ModelMode = ModelMode.TRAINING,
        models_dir: str = 'models/',
    ):
        self.storage = storage_backend
        self.mode = mode
        self.models_dir = models_dir.rstrip('/') + '/'

        logger.info(f"ModelManager initialized in {mode.value} mode")

    def save_model(
        self,
        model: nn.Module,
        model_name: str,
        metadata: Optional[Dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Save a PyTorch model with metadata.

        Parameters
        ----------
        model : nn.Module
            PyTorch model to save
        model_name : str
            Model identifier (e.g., 'wspr_HE_LE')
        metadata : dict, optional
            Model metadata (config, tickers, etc.)
        optimizer : Optimizer, optional
            Optimizer state (for resuming training)
        epoch : int, optional
            Current epoch number
        metrics : dict, optional
            Training metrics (loss, accuracy, etc.)
        overwrite : bool
            If True, overwrite existing model

        Returns
        -------
        str
            Storage path of saved model

        Raises
        ------
        FileExistsError
            If model exists and overwrite=False
        """
        # Build checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'model_class': model.__class__.__name__,
            'mode': self.mode.value,
            'timestamp': datetime.now().isoformat(),
        }

        if metadata:
            checkpoint['metadata'] = metadata

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if metrics:
            checkpoint['metrics'] = metrics

        # Determine file path
        model_filename = f"{model_name}.pth"
        model_path = self.models_dir + model_filename

        # Check if exists
        if not overwrite and self.storage.exists(model_path):
            raise FileExistsError(
                f"Model already exists: {model_path}. "
                f"Set overwrite=True to replace."
            )

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            torch.save(checkpoint, tmp_path)

        try:
            # Upload to storage
            storage_path = self.storage.upload_file(tmp_path, model_path)
            logger.info(f"Model saved: {storage_path}")

            # Save metadata separately (JSON)
            if metadata or metrics:
                self._save_metadata(model_name, checkpoint)

            return storage_path

        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

    def load_model(
        self,
        model_name: str,
        model_class: Optional[type] = None,
        device: str = 'cpu',
        strict: bool = True,
    ) -> tuple[Optional[nn.Module], Dict]:
        """
        Load a saved model.

        Parameters
        ----------
        model_name : str
            Model identifier
        model_class : type, optional
            Model class for instantiation. If None, returns state_dict only.
        device : str
            Device to load model onto
        strict : bool
            Strict state dict loading

        Returns
        -------
        model : nn.Module or None
            Loaded model (None if model_class not provided)
        metadata : dict
            Model metadata and checkpoint info

        Raises
        ------
        FileNotFoundError
            If model not found in storage
        """
        model_filename = f"{model_name}.pth"
        model_path = self.models_dir + model_filename

        if not self.storage.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Download to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.storage.download_file(model_path, tmp_path)

            # Load checkpoint
            checkpoint = torch.load(tmp_path, map_location=device)

            # Extract metadata
            metadata = {
                'model_name': checkpoint.get('model_name'),
                'model_class': checkpoint.get('model_class'),
                'mode': checkpoint.get('mode'),
                'timestamp': checkpoint.get('timestamp'),
                'epoch': checkpoint.get('epoch'),
                'metrics': checkpoint.get('metrics'),
            }

            if 'metadata' in checkpoint:
                metadata.update(checkpoint['metadata'])

            # Instantiate model if class provided
            model = None
            if model_class:
                # Try to instantiate from metadata config
                if 'config' in checkpoint.get('metadata', {}):
                    config = checkpoint['metadata']['config']
                    try:
                        model = model_class(**config)
                    except TypeError:
                        logger.warning("Could not instantiate model from config")
                        model = None

                # Load state dict
                if model:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
                    model.to(device)
                    model.eval()
                    logger.info(f"Model loaded: {model_name}")

            logger.info(f"Model checkpoint loaded: {model_name}")
            return model, metadata

        finally:
            tmp_path.unlink(missing_ok=True)

    def load_checkpoint(
        self,
        model_name: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu',
    ) -> Dict:
        """
        Load model checkpoint for resuming training.

        Parameters
        ----------
        model_name : str
            Model identifier
        model : nn.Module
            Model instance to load state into
        optimizer : Optimizer, optional
            Optimizer to load state into
        device : str
            Device to load onto

        Returns
        -------
        dict
            Checkpoint metadata (epoch, metrics, etc.)
        """
        model_filename = f"{model_name}.pth"
        model_path = self.models_dir + model_filename

        if not self.storage.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        # Download to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.storage.download_file(model_path, tmp_path)

            # Load checkpoint
            checkpoint = torch.load(tmp_path, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

            # Load optimizer state if provided
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            metadata = {
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
                'timestamp': checkpoint.get('timestamp'),
            }

            logger.info(f"Checkpoint loaded: {model_name} (epoch {metadata['epoch']})")
            return metadata

        finally:
            tmp_path.unlink(missing_ok=True)

    def list_models(self, suffix: str = '.pth') -> List[Dict[str, Any]]:
        """
        List all available models.

        Parameters
        ----------
        suffix : str
            File suffix filter

        Returns
        -------
        List[dict]
            List of model info dicts
        """
        model_files = self.storage.list_files(prefix=self.models_dir, suffix=suffix)

        models = []
        for file_path in model_files:
            try:
                metadata = self.storage.get_metadata(file_path)
                model_name = Path(file_path).stem

                # Try to load metadata JSON if available
                json_path = file_path.replace('.pth', '_metadata.json')
                if self.storage.exists(json_path):
                    meta_dict = self._load_metadata(model_name)
                else:
                    meta_dict = {}

                models.append({
                    'model_name': model_name,
                    'path': file_path,
                    'size': metadata.get('size', 0),
                    'modified': metadata.get('modified'),
                    'metadata': meta_dict,
                })

            except Exception as e:
                logger.warning(f"Could not get metadata for {file_path}: {e}")

        return sorted(models, key=lambda x: x.get('modified', ''), reverse=True)

    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model.

        Parameters
        ----------
        model_name : str
            Model identifier

        Returns
        -------
        bool
            True if deleted successfully
        """
        model_filename = f"{model_name}.pth"
        model_path = self.models_dir + model_filename

        # Delete model file
        success = self.storage.delete_file(model_path)

        # Delete metadata if exists
        json_path = model_path.replace('.pth', '_metadata.json')
        if self.storage.exists(json_path):
            self.storage.delete_file(json_path)

        if success:
            logger.info(f"Model deleted: {model_name}")

        return success

    def _save_metadata(self, model_name: str, checkpoint: Dict):
        """Save metadata as separate JSON file."""
        metadata = {
            'model_name': model_name,
            'model_class': checkpoint.get('model_class'),
            'mode': checkpoint.get('mode'),
            'timestamp': checkpoint.get('timestamp'),
            'epoch': checkpoint.get('epoch'),
            'metrics': checkpoint.get('metrics'),
        }

        if 'metadata' in checkpoint:
            metadata.update(checkpoint['metadata'])

        # Convert to JSON-serializable format
        metadata = self._make_serializable(metadata)

        # Save to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(metadata, tmp, indent=2)

        try:
            json_path = self.models_dir + f"{model_name}_metadata.json"
            self.storage.upload_file(tmp_path, json_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _load_metadata(self, model_name: str) -> Dict:
        """Load metadata from JSON file."""
        json_path = self.models_dir + f"{model_name}_metadata.json"

        if not self.storage.exists(json_path):
            return {}

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.storage.download_file(json_path, tmp_path)
            with open(tmp_path, 'r') as f:
                return json.load(f)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: ModelManager._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ModelManager._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)


def create_model_manager(
    mode: Union[str, ModelMode] = ModelMode.TRAINING,
    use_s3: bool = False,
    local_dir: Union[str, Path] = 'data/',
    s3_config: Optional[Union[S3Config, dict]] = None,
) -> ModelManager:
    """
    Convenience function to create ModelManager.

    Parameters
    ----------
    mode : str or ModelMode
        Operating mode ('training', 'backtest', or 'production')
    use_s3 : bool
        If True, use S3 storage; otherwise use local
    local_dir : str or Path
        Local storage directory (for LocalStorage)
    s3_config : S3Config or dict, optional
        S3 configuration (required if use_s3=True)

    Returns
    -------
    ModelManager
        Configured model manager

    Example
    -------
    >>> # Local training
    >>> manager = create_model_manager(mode='training', local_dir='./models')
    >>>
    >>> # S3 production
    >>> manager = create_model_manager(
    ...     mode='production',
    ...     use_s3=True,
    ...     s3_config={'bucket_name': 'ctaflow-prod'}
    ... )
    """
    # Convert mode string to enum
    if isinstance(mode, str):
        mode = ModelMode(mode.lower())

    # Create storage backend
    if use_s3:
        if s3_config is None:
            raise ValueError("s3_config required when use_s3=True")

        aws_client = AWSClient(s3_config)
        bucket_name = s3_config.bucket_name if isinstance(s3_config, S3Config) else s3_config['bucket_name']
        storage = S3Storage(bucket_name, aws_client)
    else:
        storage = LocalStorage(local_dir)

    return ModelManager(storage, mode=mode)
