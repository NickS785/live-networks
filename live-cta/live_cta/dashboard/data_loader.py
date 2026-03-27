"""
Data Loader for CTAFlow Dashboard

Unified interface for loading data from both AWS S3 and local Parquet files.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union
import logging

from .s3_client import S3Client
from .parquet_handler import ParquetHandler

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader supporting both S3 and local storage.

    Automatically handles:
    - Caching S3 downloads locally
    - Reading Parquet files efficiently
    - Loading model checkpoints and data files

    Example:
    --------
    >>> loader = DataLoader(use_s3=True, cache_dir='./cache')
    >>> # Download ticker data from S3
    >>> data_files = loader.load_ticker_data('HE')
    >>> # Load predictions
    >>> preds = loader.load_predictions('wspr_HE_LE', ticker='HE')
    """

    def __init__(
        self,
        use_s3: bool = False,
        s3_bucket: Optional[str] = None,
        cache_dir: Union[str, Path] = './data_cache',
        results_dir: Union[str, Path] = 'app/results',
    ):
        """
        Initialize data loader.

        Parameters
        ----------
        use_s3 : bool
            If True, enable S3 downloads
        s3_bucket : str, optional
            S3 bucket name (uses env var if None)
        cache_dir : str or Path
            Local directory for caching S3 downloads
        results_dir : str or Path
            Local directory for Parquet results
        """
        self.use_s3 = use_s3
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 client if enabled
        if use_s3:
            try:
                self.s3_client = S3Client(bucket_name=s3_bucket)
                logger.info("S3 client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize S3 client: {e}")
                self.s3_client = None
                self.use_s3 = False
        else:
            self.s3_client = None

        # Initialize Parquet handler for local results
        self.parquet_handler = ParquetHandler(results_dir)

    def load_ticker_data(
        self,
        ticker: str,
        data_types: Optional[List[str]] = None,
        force_download: bool = False,
    ) -> Dict[str, Path]:
        """
        Load ticker data files (features, profiles, vpin, etc.).

        Parameters
        ----------
        ticker : str
            Ticker symbol
        data_types : List[str], optional
            Specific data types to load
        force_download : bool
            If True, re-download from S3 even if cached

        Returns
        -------
        Dict[str, Path]
            Mapping of data_type -> local file path
        """
        ticker_cache = self.cache_dir / ticker

        # Check if cached locally
        if not force_download and ticker_cache.exists():
            logger.info(f"Using cached data for {ticker}")
            # Return paths to cached files
            data_files = {}
            file_map = {
                'features': 'features.csv',
                'profiles': 'profiles.npz',
                'vpin': 'vpin.parquet',
                'rasterized': 'rasterized.npz',
                'target': 'target.csv',
                'intraday': 'intraday.csv',
            }

            for data_type, filename in file_map.items():
                file_path = ticker_cache / filename
                if file_path.exists():
                    data_files[data_type] = file_path

            if data_files:
                return data_files

        # Download from S3 if enabled
        if self.use_s3 and self.s3_client:
            logger.info(f"Downloading {ticker} data from S3")
            return self.s3_client.download_ticker_data(
                ticker=ticker,
                data_types=data_types,
                local_dir=ticker_cache,
            )
        else:
            raise FileNotFoundError(
                f"No cached data for {ticker} and S3 is disabled. "
                f"Please manually place data in {ticker_cache}/"
            )

    def load_model_checkpoint(
        self,
        model_name: str,
        force_download: bool = False,
    ) -> Path:
        """
        Load model checkpoint from S3 or cache.

        Parameters
        ----------
        model_name : str
            Model identifier (e.g., 'wspr_HE_LE')
        force_download : bool
            Re-download even if cached

        Returns
        -------
        Path
            Local path to model checkpoint
        """
        model_filename = f"{model_name}.pth"
        cache_path = self.cache_dir / 'models' / model_filename
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Check cache
        if not force_download and cache_path.exists():
            logger.info(f"Using cached model: {cache_path}")
            return cache_path

        # Download from S3
        if self.use_s3 and self.s3_client:
            logger.info(f"Downloading model from S3: {model_name}")
            s3_key = f"models/{model_filename}"
            return self.s3_client.download_file(s3_key, cache_path)
        else:
            raise FileNotFoundError(
                f"Model {model_name} not found in cache and S3 is disabled. "
                f"Expected path: {cache_path}"
            )

    def list_available_models(self) -> List[Dict[str, str]]:
        """
        List all available models (S3 + local cache).

        Returns
        -------
        List[Dict[str, str]]
            List of model metadata
        """
        models = []

        # Get S3 models
        if self.use_s3 and self.s3_client:
            try:
                s3_models = self.s3_client.list_models()
                for model in s3_models:
                    model['source'] = 's3'
                models.extend(s3_models)
            except Exception as e:
                logger.warning(f"Could not list S3 models: {e}")

        # Get cached models
        cache_models_dir = self.cache_dir / 'models'
        if cache_models_dir.exists():
            for model_file in cache_models_dir.glob('*.pth'):
                models.append({
                    'key': str(model_file),
                    'name': model_file.stem,
                    'source': 'cache',
                    'size': model_file.stat().st_size,
                })

        # Deduplicate by name
        seen_names = set()
        unique_models = []
        for model in models:
            if model['name'] not in seen_names:
                unique_models.append(model)
                seen_names.add(model['name'])

        return unique_models

    def load_predictions(
        self,
        model_name: str,
        ticker: Optional[str] = None,
        date_range: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """
        Load prediction results from Parquet.

        Parameters
        ----------
        model_name : str
            Model identifier
        ticker : str, optional
            Specific ticker
        date_range : tuple, optional
            (start_date, end_date)

        Returns
        -------
        pd.DataFrame
            Predictions dataframe
        """
        return self.parquet_handler.load_predictions(
            model_name=model_name,
            ticker=ticker,
            date_range=date_range,
        )

    def save_predictions(
        self,
        df: pd.DataFrame,
        model_name: str,
        ticker: Optional[str] = None,
        upload_to_s3: bool = False,
    ) -> Path:
        """
        Save predictions to Parquet (and optionally S3).

        Parameters
        ----------
        df : pd.DataFrame
            Predictions dataframe
        model_name : str
            Model identifier
        ticker : str, optional
            Ticker symbol
        upload_to_s3 : bool
            If True, also upload to S3 after saving locally

        Returns
        -------
        Path
            Local path to saved file
        """
        # Save locally
        local_path = self.parquet_handler.save_predictions(
            df=df,
            model_name=model_name,
            ticker=ticker,
        )

        # Upload to S3 if requested
        if upload_to_s3 and self.use_s3 and self.s3_client:
            s3_key = f"predictions/{local_path.name}"
            try:
                self.s3_client.upload_file(local_path, s3_key)
                logger.info(f"Uploaded predictions to S3: {s3_key}")
            except Exception as e:
                logger.warning(f"Could not upload to S3: {e}")

        return local_path

    def load_metrics(
        self,
        model_name: str,
        split: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load training metrics from Parquet."""
        return self.parquet_handler.load_metrics(model_name, split)

    def save_metrics(
        self,
        metrics: Dict,
        model_name: str,
        split: str = 'validation',
    ) -> Path:
        """Save training metrics to Parquet."""
        return self.parquet_handler.save_metrics(metrics, model_name, split)


# Convenience function
def get_data_loader(**kwargs) -> DataLoader:
    """Create DataLoader instance."""
    return DataLoader(**kwargs)
