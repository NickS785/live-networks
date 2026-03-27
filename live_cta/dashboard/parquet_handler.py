"""
Parquet Data Handler for CTAFlow Results Storage

Handles saving and loading prediction results, backtest outputs,
and performance metrics in Parquet format for efficient storage and retrieval.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, List, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ParquetHandler:
    """
    Handler for reading/writing CTAFlow results in Parquet format.

    Parquet provides:
    - Efficient columnar storage
    - Built-in compression
    - Schema enforcement
    - Fast filtering and aggregation

    Example:
    --------
    >>> handler = ParquetHandler('app/results')
    >>> handler.save_predictions(predictions_df, model_name='wspr_HE_LE')
    >>> df = handler.load_predictions(model_name='wspr_HE_LE', date_range=('2024-01-01', '2024-12-31'))
    """

    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize Parquet handler.

        Parameters
        ----------
        results_dir : str or Path
            Root directory for storing results
        """
        self.results_dir = Path(results_dir)
        self.predictions_dir = self.results_dir / 'predictions'
        self.backtests_dir = self.results_dir / 'backtests'
        self.metrics_dir = self.results_dir / 'metrics'

        # Create directories
        for dir_path in [self.predictions_dir, self.backtests_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_predictions(
        self,
        df: pd.DataFrame,
        model_name: str,
        ticker: Optional[str] = None,
        compression: str = 'snappy',
    ) -> Path:
        """
        Save model predictions to Parquet.

        Expected DataFrame columns:
        - datetime: Timestamp
        - ticker: str (if ticker=None)
        - prediction: float or int (model output)
        - probability_*: float (class probabilities for classification)
        - target: float or int (actual target value, if available)
        - pnl: float (realized PnL, if available)

        Parameters
        ----------
        df : pd.DataFrame
            Predictions dataframe
        model_name : str
            Model identifier
        ticker : str, optional
            Specific ticker (if None, assumes multi-ticker df)
        compression : str
            Compression algorithm ('snappy', 'gzip', 'brotli')

        Returns
        -------
        Path
            Path to saved Parquet file
        """
        if ticker:
            file_path = self.predictions_dir / f"{model_name}_{ticker}.parquet"
        else:
            file_path = self.predictions_dir / f"{model_name}_all.parquet"

        logger.info(f"Saving predictions to {file_path}")

        # Ensure datetime column
        if 'datetime' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])

        # Write to Parquet
        df.to_parquet(
            file_path,
            engine='pyarrow',
            compression=compression,
            index=False,
        )

        logger.info(f"Saved {len(df)} predictions to {file_path}")
        return file_path

    def load_predictions(
        self,
        model_name: str,
        ticker: Optional[str] = None,
        date_range: Optional[tuple] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load predictions from Parquet.

        Parameters
        ----------
        model_name : str
            Model identifier
        ticker : str, optional
            Specific ticker
        date_range : tuple, optional
            (start_date, end_date) for filtering
        columns : List[str], optional
            Specific columns to load

        Returns
        -------
        pd.DataFrame
            Predictions dataframe
        """
        if ticker:
            file_path = self.predictions_dir / f"{model_name}_{ticker}.parquet"
        else:
            file_path = self.predictions_dir / f"{model_name}_all.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {file_path}")

        logger.info(f"Loading predictions from {file_path}")

        # Build filters for efficient reading
        filters = []
        if date_range:
            start_date, end_date = date_range
            filters.append(('datetime', '>=', pd.Timestamp(start_date)))
            filters.append(('datetime', '<=', pd.Timestamp(end_date)))

        # Read Parquet with filters
        df = pd.read_parquet(
            file_path,
            engine='pyarrow',
            columns=columns,
            filters=filters if filters else None,
        )

        logger.info(f"Loaded {len(df)} predictions")
        return df

    def save_backtest_results(
        self,
        df: pd.DataFrame,
        model_name: str,
        ticker: str,
        strategy_name: str,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """
        Save backtest results to Parquet.

        Expected DataFrame columns:
        - datetime: Timestamp
        - position: int (-1, 0, 1)
        - pnl: float
        - cumulative_pnl: float
        - sharpe: float (optional)
        - drawdown: float (optional)

        Parameters
        ----------
        df : pd.DataFrame
            Backtest results
        model_name : str
            Model identifier
        ticker : str
            Ticker symbol
        strategy_name : str
            Strategy identifier
        metadata : dict, optional
            Additional metadata to store

        Returns
        -------
        Path
            Path to saved Parquet file
        """
        file_name = f"{model_name}_{ticker}_{strategy_name}_{datetime.now():%Y%m%d_%H%M%S}.parquet"
        file_path = self.backtests_dir / file_name

        logger.info(f"Saving backtest results to {file_path}")

        # Add metadata as columns if provided
        if metadata:
            for key, value in metadata.items():
                df[f'meta_{key}'] = value

        # Write to Parquet
        df.to_parquet(
            file_path,
            engine='pyarrow',
            compression='snappy',
            index=False,
        )

        logger.info(f"Saved backtest with {len(df)} rows to {file_path}")
        return file_path

    def save_metrics(
        self,
        metrics: Dict,
        model_name: str,
        split: str = 'validation',
    ) -> Path:
        """
        Save training/validation metrics to Parquet.

        Parameters
        ----------
        metrics : dict
            Metrics dictionary (e.g., from TrainingMetrics)
        model_name : str
            Model identifier
        split : str
            Data split ('train', 'validation', 'test')

        Returns
        -------
        Path
            Path to saved Parquet file
        """
        file_path = self.metrics_dir / f"{model_name}_{split}_metrics.parquet"

        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame([metrics])
        metrics_df['timestamp'] = datetime.now()
        metrics_df['model_name'] = model_name
        metrics_df['split'] = split

        # Append if file exists
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)

        # Write to Parquet
        metrics_df.to_parquet(
            file_path,
            engine='pyarrow',
            compression='snappy',
            index=False,
        )

        logger.info(f"Saved metrics to {file_path}")
        return file_path

    def load_metrics(
        self,
        model_name: str,
        split: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load metrics from Parquet.

        Parameters
        ----------
        model_name : str
            Model identifier
        split : str, optional
            Specific split to load. If None, loads all splits.

        Returns
        -------
        pd.DataFrame
            Metrics dataframe
        """
        if split:
            file_path = self.metrics_dir / f"{model_name}_{split}_metrics.parquet"
            if not file_path.exists():
                raise FileNotFoundError(f"Metrics file not found: {file_path}")
            return pd.read_parquet(file_path)
        else:
            # Load all splits
            pattern = f"{model_name}_*_metrics.parquet"
            files = list(self.metrics_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No metrics files found for model: {model_name}")

            dfs = [pd.read_parquet(f) for f in files]
            return pd.concat(dfs, ignore_index=True)

    def list_available_predictions(self) -> List[Dict[str, str]]:
        """
        List all available prediction files.

        Returns
        -------
        List[Dict[str, str]]
            List of dicts with 'model_name', 'ticker', 'file_path'
        """
        files = list(self.predictions_dir.glob("*.parquet"))

        predictions = []
        for file_path in files:
            name_parts = file_path.stem.split('_')
            if len(name_parts) >= 2:
                model_name = '_'.join(name_parts[:-1])
                ticker = name_parts[-1] if name_parts[-1] != 'all' else 'multi'
            else:
                model_name = file_path.stem
                ticker = 'unknown'

            predictions.append({
                'model_name': model_name,
                'ticker': ticker,
                'file_path': str(file_path),
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            })

        return predictions


# Convenience function
def get_parquet_handler(results_dir: Union[str, Path] = 'app/results') -> ParquetHandler:
    """Get or create ParquetHandler instance."""
    return ParquetHandler(results_dir)
