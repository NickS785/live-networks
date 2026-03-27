"""
Storage Backend Abstraction for CTAFlow

Provides pluggable storage backends for local and cloud storage.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Union, BinaryIO
import logging

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Implementations must support:
    - File upload/download
    - Directory listing
    - File existence checks
    - File metadata retrieval
    """

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a file/directory exists."""
        pass

    @abstractmethod
    def upload_file(self, local_path: Union[str, Path], remote_path: str) -> str:
        """Upload a file to storage."""
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        """Download a file from storage."""
        pass

    @abstractmethod
    def list_files(self, prefix: str = '', suffix: str = '') -> List[str]:
        """List files matching prefix/suffix."""
        pass

    @abstractmethod
    def delete_file(self, path: str) -> bool:
        """Delete a file."""
        pass

    @abstractmethod
    def get_metadata(self, path: str) -> Dict[str, any]:
        """Get file metadata (size, modified time, etc.)."""
        pass


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend.

    Parameters
    ----------
    root_dir : str or Path
        Root directory for local storage

    Example
    -------
    >>> storage = LocalStorage('data/')
    >>> storage.upload_file('model.pth', 'models/my_model.pth')
    >>> storage.exists('models/my_model.pth')
    True
    """

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized at {self.root_dir}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve a storage path to absolute filesystem path."""
        return self.root_dir / path

    def exists(self, path: str) -> bool:
        """Check if file exists."""
        return self._resolve_path(path).exists()

    def upload_file(self, local_path: Union[str, Path], remote_path: str) -> str:
        """
        Copy file to storage location.

        Parameters
        ----------
        local_path : str or Path
            Source file path
        remote_path : str
            Destination path within storage

        Returns
        -------
        str
            Stored file path
        """
        local_path = Path(local_path)
        dest_path = self._resolve_path(remote_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists():
            raise FileNotFoundError(f"Source file not found: {local_path}")

        import shutil
        shutil.copy2(local_path, dest_path)
        logger.info(f"Uploaded {local_path} -> {dest_path}")

        return str(dest_path)

    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        """
        Copy file from storage to local path.

        Parameters
        ----------
        remote_path : str
            Source path in storage
        local_path : str or Path
            Destination local path

        Returns
        -------
        Path
            Local file path
        """
        source_path = self._resolve_path(remote_path)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not source_path.exists():
            raise FileNotFoundError(f"Storage file not found: {remote_path}")

        import shutil
        shutil.copy2(source_path, local_path)
        logger.info(f"Downloaded {source_path} -> {local_path}")

        return local_path

    def list_files(self, prefix: str = '', suffix: str = '') -> List[str]:
        """
        List files matching prefix and suffix.

        Parameters
        ----------
        prefix : str
            Path prefix filter (e.g., 'models/')
        suffix : str
            File extension filter (e.g., '.pth')

        Returns
        -------
        List[str]
            List of matching file paths (relative to root_dir)
        """
        search_dir = self.root_dir / prefix if prefix else self.root_dir

        if not search_dir.exists():
            logger.warning(f"Directory not found: {search_dir}")
            return []

        # Recursively find matching files
        pattern = f"**/*{suffix}" if suffix else "**/*"
        files = []

        for file_path in search_dir.glob(pattern):
            if file_path.is_file():
                # Get path relative to root_dir
                rel_path = file_path.relative_to(self.root_dir)
                files.append(str(rel_path))

        logger.debug(f"Found {len(files)} files with prefix='{prefix}', suffix='{suffix}'")
        return sorted(files)

    def delete_file(self, path: str) -> bool:
        """
        Delete a file.

        Parameters
        ----------
        path : str
            File path to delete

        Returns
        -------
        bool
            True if deleted, False if file didn't exist
        """
        file_path = self._resolve_path(path)

        if not file_path.exists():
            logger.warning(f"File not found for deletion: {path}")
            return False

        file_path.unlink()
        logger.info(f"Deleted file: {path}")
        return True

    def get_metadata(self, path: str) -> Dict[str, any]:
        """
        Get file metadata.

        Parameters
        ----------
        path : str
            File path

        Returns
        -------
        dict
            Metadata including size, modified time, etc.
        """
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stat = file_path.stat()

        return {
            'path': str(path),
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'is_file': file_path.is_file(),
            'is_dir': file_path.is_dir(),
        }


class S3Storage(StorageBackend):
    """
    AWS S3 storage backend.

    Parameters
    ----------
    bucket_name : str
        S3 bucket name
    aws_client : AWSClient
        Configured AWS client instance
    prefix : str, optional
        Optional prefix for all paths (e.g., 'ctaflow/')

    Example
    -------
    >>> from CTAFlow.data.storage import AWSClient
    >>> aws = AWSClient(bucket='my-bucket')
    >>> storage = S3Storage(bucket_name='my-bucket', aws_client=aws)
    >>> storage.upload_file('model.pth', 'models/my_model.pth')
    """

    def __init__(self, bucket_name: str, aws_client, prefix: str = ''):
        self.bucket_name = bucket_name
        self.aws_client = aws_client
        self.prefix = prefix.rstrip('/') + '/' if prefix else ''
        logger.info(f"S3Storage initialized for bucket: {bucket_name}, prefix: {prefix}")

    def _add_prefix(self, path: str) -> str:
        """Add storage prefix to path."""
        return self.prefix + path.lstrip('/')

    def exists(self, path: str) -> bool:
        """Check if object exists in S3."""
        s3_key = self._add_prefix(path)
        return self.aws_client.object_exists(s3_key)

    def upload_file(self, local_path: Union[str, Path], remote_path: str) -> str:
        """Upload file to S3."""
        local_path = Path(local_path)
        s3_key = self._add_prefix(remote_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        self.aws_client.upload_file(str(local_path), s3_key)
        return f"s3://{self.bucket_name}/{s3_key}"

    def download_file(self, remote_path: str, local_path: Union[str, Path]) -> Path:
        """Download file from S3."""
        s3_key = self._add_prefix(remote_path)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        self.aws_client.download_file(s3_key, str(local_path))
        return local_path

    def list_files(self, prefix: str = '', suffix: str = '') -> List[str]:
        """List S3 objects matching prefix/suffix."""
        s3_prefix = self._add_prefix(prefix)
        all_keys = self.aws_client.list_objects(s3_prefix)

        # Filter by suffix
        if suffix:
            all_keys = [k for k in all_keys if k.endswith(suffix)]

        # Remove storage prefix from results
        if self.prefix:
            all_keys = [k[len(self.prefix):] if k.startswith(self.prefix) else k
                       for k in all_keys]

        return sorted(all_keys)

    def delete_file(self, path: str) -> bool:
        """Delete S3 object."""
        s3_key = self._add_prefix(path)
        return self.aws_client.delete_object(s3_key)

    def get_metadata(self, path: str) -> Dict[str, any]:
        """Get S3 object metadata."""
        s3_key = self._add_prefix(path)
        return self.aws_client.get_object_metadata(s3_key)
