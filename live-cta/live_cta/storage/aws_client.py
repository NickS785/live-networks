"""
AWS Client for CTAFlow Data Management

Handles all AWS S3 interactions for:
- Downloading training data
- Uploading models and predictions
- Managing data lifecycle
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Union
from dataclasses import dataclass
import logging

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    """
    Configuration for S3 client.

    Parameters
    ----------
    bucket_name : str
        S3 bucket name
    region : str
        AWS region (default: 'us-east-1')
    endpoint_url : str, optional
        Custom S3 endpoint URL (for S3-compatible services)
    access_key_id : str, optional
        AWS access key (uses env var if None)
    secret_access_key : str, optional
        AWS secret key (uses env var if None)
    prefix : str, optional
        Default prefix for all operations (e.g., 'ctaflow/')

    Example
    -------
    >>> config = S3Config(
    ...     bucket_name='my-data-bucket',
    ...     region='us-west-2',
    ...     prefix='ctaflow/'
    ... )
    """
    bucket_name: str
    region: str = 'us-east-1'
    endpoint_url: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    prefix: str = ''


class AWSClient:
    """
    AWS S3 client for CTAFlow data operations.

    Provides high-level interface for S3 operations with:
    - Automatic credential handling
    - Custom endpoint support (S3-compatible services)
    - Bulk operations
    - Error handling and retries

    Parameters
    ----------
    config : S3Config or dict
        S3 configuration
    cache_dir : str or Path, optional
        Local cache directory for downloads

    Example
    -------
    >>> # Using environment variables
    >>> client = AWSClient(S3Config(bucket_name='my-bucket'))
    >>>
    >>> # Using explicit credentials
    >>> config = S3Config(
    ...     bucket_name='my-bucket',
    ...     access_key_id='KEY',
    ...     secret_access_key='SECRET',
    ...     endpoint_url='https://s3.custom.com'  # Optional
    ... )
    >>> client = AWSClient(config)
    >>>
    >>> # Download data
    >>> client.download_ticker_data('HE', local_dir='data/HE')
    >>>
    >>> # Upload model
    >>> client.upload_file('model.pth', 'models/my_model.pth')
    """

    def __init__(
        self,
        config: Union[S3Config, dict],
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for AWS operations. "
                "Install with: pip install boto3"
            )

        # Convert dict to S3Config if needed
        if isinstance(config, dict):
            config = S3Config(**config)

        self.config = config
        self.bucket_name = config.bucket_name
        self.prefix = config.prefix

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / '.ctaflow' / 'cache' / 's3'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build boto3 client configuration
        client_kwargs = {'region_name': config.region}

        if config.endpoint_url:
            client_kwargs['endpoint_url'] = config.endpoint_url

        if config.access_key_id and config.secret_access_key:
            client_kwargs['aws_access_key_id'] = config.access_key_id
            client_kwargs['aws_secret_access_key'] = config.secret_access_key
        else:
            # Use environment variables or AWS config
            # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
            pass

        try:
            self.s3_client = boto3.client('s3', **client_kwargs)
            self.s3_resource = boto3.resource('s3', **client_kwargs)
            logger.info(f"AWS S3 client initialized for bucket: {self.bucket_name}")

            # Test connection
            self._test_connection()

        except NoCredentialsError:
            logger.error(
                "AWS credentials not found. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables, or provide "
                "them in S3Config."
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def _test_connection(self):
        """Test S3 connection by checking bucket access."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.warning(f"Bucket not found: {self.bucket_name}")
            elif error_code == '403':
                logger.error(f"Access denied to bucket: {self.bucket_name}")
                raise
            else:
                logger.error(f"Error accessing bucket: {e}")
                raise

    def _add_prefix(self, key: str) -> str:
        """Add default prefix to S3 key."""
        if self.prefix:
            return self.prefix.rstrip('/') + '/' + key.lstrip('/')
        return key

    def object_exists(self, key: str) -> bool:
        """Check if S3 object exists."""
        key = self._add_prefix(key)
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError:
            return False

    def upload_file(
        self,
        local_path: Union[str, Path],
        s3_key: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Upload a file to S3.

        Parameters
        ----------
        local_path : str or Path
            Local file path
        s3_key : str
            S3 object key (path in bucket)
        metadata : dict, optional
            Object metadata

        Returns
        -------
        str
            S3 URI (s3://bucket/key)
        """
        local_path = Path(local_path)
        s3_key = self._add_prefix(s3_key)

        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        extra_args = {}
        if metadata:
            extra_args['Metadata'] = metadata

        try:
            logger.info(f"Uploading {local_path} -> s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.info(f"Upload complete: {s3_uri}")
            return s3_uri

        except ClientError as e:
            logger.error(f"Upload failed: {e}")
            raise

    def download_file(
        self,
        s3_key: str,
        local_path: Union[str, Path],
        use_cache: bool = True,
    ) -> Path:
        """
        Download a file from S3.

        Parameters
        ----------
        s3_key : str
            S3 object key
        local_path : str or Path
            Local destination path
        use_cache : bool
            If True, check cache first

        Returns
        -------
        Path
            Local file path
        """
        s3_key = self._add_prefix(s3_key)
        local_path = Path(local_path)

        # Check cache
        if use_cache:
            cache_path = self.cache_dir / s3_key
            if cache_path.exists():
                logger.info(f"Using cached file: {cache_path}")
                # Copy from cache to destination
                import shutil
                local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cache_path, local_path)
                return local_path

        # Download from S3
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} -> {local_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(f"Download complete: {local_path}")

            # Update cache
            if use_cache:
                cache_path = self.cache_dir / s3_key
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(local_path, cache_path)

            return local_path

        except ClientError as e:
            logger.error(f"Download failed: {e}")
            raise

    def list_objects(self, prefix: str = '', suffix: str = '') -> List[str]:
        """
        List S3 objects matching prefix and suffix.

        Parameters
        ----------
        prefix : str
            Key prefix filter
        suffix : str
            Key suffix filter (e.g., '.pth')

        Returns
        -------
        List[str]
            List of matching S3 keys
        """
        prefix = self._add_prefix(prefix)

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            keys = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if not suffix or key.endswith(suffix):
                            keys.append(key)

            logger.info(f"Found {len(keys)} objects with prefix='{prefix}', suffix='{suffix}'")
            return keys

        except ClientError as e:
            logger.error(f"List objects failed: {e}")
            raise

    def delete_object(self, s3_key: str) -> bool:
        """Delete an S3 object."""
        s3_key = self._add_prefix(s3_key)

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Delete failed: {e}")
            return False

    def get_object_metadata(self, s3_key: str) -> Dict[str, any]:
        """Get S3 object metadata."""
        s3_key = self._add_prefix(s3_key)

        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                'path': s3_key,
                'size': response['ContentLength'],
                'modified': response['LastModified'].isoformat(),
                'etag': response['ETag'],
                'metadata': response.get('Metadata', {}),
            }
        except ClientError as e:
            logger.error(f"Get metadata failed: {e}")
            raise

    def download_directory(
        self,
        s3_prefix: str,
        local_dir: Union[str, Path],
        use_cache: bool = True,
    ) -> Dict[str, Path]:
        """
        Recursively download all objects under an S3 prefix.

        Parameters
        ----------
        s3_prefix : str
            S3 prefix to download (e.g., 'workspace/model_data/GC/')
        local_dir : str or Path
            Local destination directory
        use_cache : bool
            If True, check cache before downloading

        Returns
        -------
        Dict[str, Path]
            Mapping of S3 key -> local file path for all downloaded files
        """
        full_prefix = self._add_prefix(s3_prefix)
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.bucket_name, Prefix=full_prefix
            )

            downloaded = {}
            for page in pages:
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    key = obj['Key']
                    # Skip "directory" markers (zero-byte keys ending in /)
                    if key.endswith('/'):
                        continue
                    # Compute relative path from the prefix
                    relative = key[len(full_prefix):].lstrip('/')
                    if not relative:
                        continue
                    local_path = local_dir / relative
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download using raw key (already has prefix)
                    if use_cache:
                        cache_path = self.cache_dir / key
                        if cache_path.exists():
                            logger.info(f"Using cached file: {cache_path}")
                            import shutil
                            shutil.copy2(cache_path, local_path)
                            downloaded[key] = local_path
                            continue

                    try:
                        logger.info(
                            f"Downloading s3://{self.bucket_name}/{key} "
                            f"-> {local_path}"
                        )
                        self.s3_client.download_file(
                            self.bucket_name, key, str(local_path)
                        )
                        downloaded[key] = local_path

                        if use_cache:
                            cache_path = self.cache_dir / key
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            import shutil
                            shutil.copy2(local_path, cache_path)

                    except ClientError as e:
                        logger.warning(f"Failed to download {key}: {e}")

            logger.info(
                f"Downloaded {len(downloaded)} files from "
                f"s3://{self.bucket_name}/{full_prefix}"
            )
            return downloaded

        except ClientError as e:
            logger.error(f"Directory download failed: {e}")
            raise

    def download_ticker_data(
        self,
        ticker: str,
        data_types: Optional[List[str]] = None,
        local_dir: Optional[Union[str, Path]] = None,
        s3_prefix: str = 'workspace/model_data/',
    ) -> Dict[str, Path]:
        """
        Download all data files for a ticker.

        Expected S3 structure:
        s3://bucket/workspace/model_data/{ticker}/features.csv
        s3://bucket/workspace/model_data/{ticker}/profiles.npz
        s3://bucket/workspace/model_data/{ticker}/vpin.parquet
        etc.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        data_types : List[str], optional
            Specific data types to download:
            ['features', 'profiles', 'vpin', 'rasterized', 'target', 'intraday']
        local_dir : str or Path, optional
            Local directory (default: cache_dir/data/{ticker})
        s3_prefix : str
            S3 prefix for ticker data (default: 'data/')

        Returns
        -------
        Dict[str, Path]
            Mapping of data_type -> local file path
        """
        if data_types is None:
            data_types = ['features', 'profiles', 'vpin', 'rasterized', 'target', 'intraday']

        if local_dir is None:
            local_dir = self.cache_dir / 'data' / ticker
        else:
            local_dir = Path(local_dir)

        local_dir.mkdir(parents=True, exist_ok=True)

        # File mappings
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

            s3_key = f"{s3_prefix.rstrip('/')}/{ticker}/{file_map[data_type]}"
            local_path = local_dir / file_map[data_type]

            try:
                self.download_file(s3_key, local_path)
                downloaded[data_type] = local_path
            except ClientError as e:
                logger.warning(f"Could not download {data_type} for {ticker}: {e}")

        logger.info(f"Downloaded {len(downloaded)} data files for {ticker}")
        return downloaded

    def list_tickers(self, data_prefix: str = 'workspace/model_data/') -> List[str]:
        """
        List available tickers in S3.

        Parameters
        ----------
        data_prefix : str
            S3 prefix for ticker directories

        Returns
        -------
        List[str]
            List of ticker symbols
        """
        prefix = self._add_prefix(data_prefix)

        try:
            # List all objects with prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                Delimiter='/'
            )

            tickers = []
            if 'CommonPrefixes' in response:
                for prefix_obj in response['CommonPrefixes']:
                    # Extract ticker from prefix (e.g., 'data/HE/' -> 'HE')
                    ticker_prefix = prefix_obj['Prefix']
                    ticker = ticker_prefix.rstrip('/').split('/')[-1]
                    tickers.append(ticker)

            logger.info(f"Found {len(tickers)} tickers in S3")
            return sorted(tickers)

        except ClientError as e:
            logger.error(f"List tickers failed: {e}")
            raise


def create_s3_client(
    bucket_name: str,
    region: str = 'us-east-1',
    endpoint_url: Optional[str] = None,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    prefix: str = '',
) -> AWSClient:
    """
    Convenience function to create AWSClient.

    Parameters
    ----------
    bucket_name : str
        S3 bucket name
    region : str
        AWS region
    endpoint_url : str, optional
        Custom S3 endpoint
    access_key_id : str, optional
        AWS access key
    secret_access_key : str, optional
        AWS secret key
    prefix : str, optional
        Default prefix

    Returns
    -------
    AWSClient
        Configured AWS client
    """
    config = S3Config(
        bucket_name=bucket_name,
        region=region,
        endpoint_url=endpoint_url,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        prefix=prefix,
    )

    return AWSClient(config)
