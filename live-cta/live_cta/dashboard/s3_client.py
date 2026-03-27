"""
AWS S3 Client for CTAFlow Data and Model Management

Handles downloading and uploading:
- Trained model checkpoints
- Training/validation data
- Prediction results
- Backtest outputs
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Union
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import logging

logger = logging.getLogger(__name__)


class S3Client:
    """
    Client for interacting with AWS S3 for CTAFlow data storage.

    Environment Variables:
    ---------------------
    AWS_ACCESS_KEY_ID : str
        AWS access key ID
    AWS_SECRET_ACCESS_KEY : str
        AWS secret access key
    AWS_DEFAULT_REGION : str (optional)
        AWS region (default: 'us-east-1')
    CTAFLOW_S3_BUCKET : str (optional)
        Default S3 bucket name (default: 'ctaflow-data')

    Example:
    --------
    >>> s3 = S3Client()
    >>> s3.download_model('models/multi_asset_wspr_HE_LE.pth', 'local/path/')
    >>> data = s3.load_parquet('predictions/latest.parquet')
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """
        Initialize S3 client.

        Parameters
        ----------
        bucket_name : str, optional
            S3 bucket name. If None, uses CTAFLOW_S3_BUCKET env var or 'ctaflow-data'
        aws_access_key_id : str, optional
            AWS access key. If None, uses AWS_ACCESS_KEY_ID env var
        aws_secret_access_key : str, optional
            AWS secret key. If None, uses AWS_SECRET_ACCESS_KEY env var
        region_name : str, optional
            AWS region. If None, uses AWS_DEFAULT_REGION env var or 'us-east-1'
        """
        self.bucket_name = bucket_name or os.getenv('CTAFLOW_S3_BUCKET', 'ctaflow-data')

        # Initialize boto3 client
        session_kwargs = {}
        if aws_access_key_id:
            session_kwargs['aws_access_key_id'] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs['aws_secret_access_key'] = aws_secret_access_key
        if region_name:
            session_kwargs['region_name'] = region_name
        else:
            session_kwargs['region_name'] = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

        try:
            self.s3_client = boto3.client('s3', **session_kwargs)
            self.s3_resource = boto3.resource('s3', **session_kwargs)
            logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            raise

    def list_files(self, prefix: str = '', suffix: str = '') -> List[str]:
        """
        List files in S3 bucket with optional prefix and suffix filters.

        Parameters
        ----------
        prefix : str
            Filter by prefix (e.g., 'models/')
        suffix : str
            Filter by suffix (e.g., '.pth')

        Returns
        -------
        List[str]
            List of S3 keys matching the filters
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                logger.warning(f"No files found with prefix: {prefix}")
                return []

            files = [obj['Key'] for obj in response['Contents']]

            if suffix:
                files = [f for f in files if f.endswith(suffix)]

            logger.info(f"Found {len(files)} files with prefix='{prefix}', suffix='{suffix}'")
            return files

        except ClientError as e:
            logger.error(f"Error listing files: {e}")
            raise

    def download_file(self, s3_key: str, local_path: Union[str, Path]) -> Path:
        """
        Download a file from S3 to local storage.

        Parameters
        ----------
        s3_key : str
            S3 object key (path in bucket)
        local_path : str or Path
            Local destination path

        Returns
        -------
        Path
            Path to the downloaded file
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} -> {local_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(f"Download complete: {local_path}")
            return local_path

        except ClientError as e:
            logger.error(f"Error downloading file: {e}")
            raise

    def upload_file(self, local_path: Union[str, Path], s3_key: str) -> str:
        """
        Upload a file from local storage to S3.

        Parameters
        ----------
        local_path : str or Path
            Local file path
        s3_key : str
            Destination S3 object key

        Returns
        -------
        str
            S3 URI (s3://bucket/key)
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            logger.info(f"Uploading {local_path} -> s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Upload complete: {s3_uri}")
            return s3_uri

        except ClientError as e:
            logger.error(f"Error uploading file: {e}")
            raise

    def load_parquet(self, s3_key: str, download: bool = True) -> Union[str, Path]:
        """
        Load a Parquet file from S3.

        Parameters
        ----------
        s3_key : str
            S3 object key for the Parquet file
        download : bool
            If True, download to temp location and return path
            If False, return S3 URI for direct reading (requires s3fs)

        Returns
        -------
        str or Path
            Local path (if download=True) or S3 URI (if download=False)
        """
        if download:
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / 'ctaflow_s3_cache'
            temp_dir.mkdir(exist_ok=True)
            local_path = temp_dir / Path(s3_key).name
            return self.download_file(s3_key, local_path)
        else:
            return f"s3://{self.bucket_name}/{s3_key}"

    def list_models(self) -> List[Dict[str, str]]:
        """
        List available trained models in the S3 bucket.

        Returns
        -------
        List[Dict[str, str]]
            List of model metadata dicts with 'key', 'name', 'size', 'modified'
        """
        model_files = self.list_files(prefix='models/', suffix='.pth')

        models = []
        for key in model_files:
            try:
                obj = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
                models.append({
                    'key': key,
                    'name': Path(key).stem,
                    'size': obj['ContentLength'],
                    'modified': obj['LastModified'].isoformat(),
                })
            except ClientError:
                continue

        return models

    def download_ticker_data(
        self,
        ticker: str,
        data_types: Optional[List[str]] = None,
        local_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """
        Download all data files for a specific ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g., 'HE', 'LE')
        data_types : List[str], optional
            Specific data types to download. Options:
            ['features', 'profiles', 'vpin', 'rasterized', 'target', 'intraday']
            If None, downloads all types
        local_dir : str or Path, optional
            Local directory for downloads. If None, uses './data/{ticker}/'

        Returns
        -------
        Dict[str, Path]
            Mapping of data_type -> local_path for downloaded files
        """
        if data_types is None:
            data_types = ['features', 'profiles', 'vpin', 'rasterized', 'target', 'intraday']

        if local_dir is None:
            local_dir = Path('data') / ticker
        else:
            local_dir = Path(local_dir)

        local_dir.mkdir(parents=True, exist_ok=True)

        # Define file mappings
        file_map = {
            'features': 'features.csv',
            'profiles': 'profiles.npz',
            'vpin': 'vpin.parquet',
            'rasterized': 'rasterized.npz',
            'target': 'target.csv',
            'intraday': 'intraday.csv',
        }

        downloaded = {}
        for data_type in data_types:
            if data_type not in file_map:
                logger.warning(f"Unknown data type: {data_type}")
                continue

            s3_key = f"data/{ticker}/{file_map[data_type]}"
            local_path = local_dir / file_map[data_type]

            try:
                self.download_file(s3_key, local_path)
                downloaded[data_type] = local_path
            except ClientError as e:
                logger.warning(f"Could not download {data_type} for {ticker}: {e}")

        return downloaded


# Convenience function for easy import
def get_s3_client(**kwargs) -> S3Client:
    """Get or create S3 client instance."""
    return S3Client(**kwargs)
