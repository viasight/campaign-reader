import os
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class CampaignZipError(Exception):
    """Base exception for campaign zip file errors."""
    pass

class CampaignReader:
    """A class for reading and analyzing campaign zip files."""
    
    def __init__(self, zip_path: str, extract_dir: Optional[str] = None):
        """Initialize the CampaignReader with a path to a zip file.
        
        Args:
            zip_path (str): Path to the campaign zip file
            extract_dir (Optional[str]): Directory to extract files to. If None, uses a temp directory
            
        Raises:
            FileNotFoundError: If zip_path doesn't exist
            CampaignZipError: If zip file is invalid or extraction fails
        """
        self.zip_path = Path(zip_path)
        self._validate_zip_file()
        
        # Set up extraction directory
        self._using_temp_dir = extract_dir is None
        self._extract_dir = Path(extract_dir) if extract_dir else Path(tempfile.mkdtemp())
        self._extracted_files: Dict[str, Path] = {}
        
        # Extract contents
        self._extract_contents()
    
    def _validate_zip_file(self) -> None:
        """Validate that the zip file exists and can be opened.
        
        Raises:
            FileNotFoundError: If file doesn't exist
            CampaignZipError: If file is not a valid zip file
        """
        if not self.zip_path.exists():
            raise FileNotFoundError(f"File {self.zip_path} not found")
            
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                # Check for zip file corruption
                if zf.testzip() is not None:
                    raise CampaignZipError(f"Zip file {self.zip_path} is corrupted")
                
                # Store file listing
                self.file_list = zf.namelist()
                
                # Basic security check for zip slip
                for fname in self.file_list:
                    if os.path.isabs(fname) or fname.startswith('..'):
                        raise CampaignZipError(f"Potentially malicious path in zip: {fname}")
                        
        except zipfile.BadZipFile:
            raise CampaignZipError(f"File {self.zip_path} is not a valid zip file")
    
    def _extract_contents(self) -> None:
        """Extract the contents of the zip file to the extraction directory.
        
        Raises:
            CampaignZipError: If extraction fails
        """
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                # Create extraction directory if it doesn't exist
                self._extract_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract each file safely
                for fname in self.file_list:
                    # Create safe extraction path
                    extract_path = self._extract_dir / fname
                    
                    # Ensure extraction path is within extract_dir
                    if not extract_path.resolve().is_relative_to(self._extract_dir.resolve()):
                        raise CampaignZipError(f"Attempted path traversal: {fname}")
                    
                    # Create parent directories if needed
                    extract_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extract and store the path
                    zf.extract(fname, self._extract_dir)
                    self._extracted_files[fname] = extract_path
                    
        except Exception as e:
            # Clean up any partially extracted files
            self.cleanup()
            raise CampaignZipError(f"Failed to extract zip contents: {str(e)}")
    
    def get_extracted_path(self, filename: str) -> Optional[Path]:
        """Get the extracted path for a given filename.
        
        Args:
            filename (str): Name of file in the zip
            
        Returns:
            Optional[Path]: Path to extracted file or None if not found
        """
        return self._extracted_files.get(filename)
    
    def list_files(self) -> List[str]:
        """Get a list of all files in the campaign zip.
        
        Returns:
            List[str]: List of filenames
        """
        return self.file_list
    
    def cleanup(self) -> None:
        """Remove all extracted files and directories."""
        if self._extract_dir.exists():
            try:
                # If directory is readonly, try to make it writable first
                current_mode = self._extract_dir.stat().st_mode
                if not os.access(self._extract_dir, os.W_OK):
                    os.chmod(self._extract_dir, current_mode | 0o700)

                # Remove files
                for file_path in self._extracted_files.values():
                    if file_path.exists():
                        file_path.unlink(missing_ok=True)
                
                # Remove empty directories, but only if we created the temp dir
                if self._using_temp_dir:
                    for dir_path in sorted(self._extract_dir.rglob('*'), reverse=True):
                        if dir_path.is_dir():
                            try:
                                dir_path.rmdir()
                            except OSError:
                                pass  # Directory might not be empty or might be readonly
                    
                    # Remove the temp directory itself
                    try:
                        self._extract_dir.rmdir()
                    except OSError:
                        pass  # Directory might not be empty or might be readonly

                # Restore original permissions if we changed them
                if not os.access(self._extract_dir, os.W_OK):
                    os.chmod(self._extract_dir, current_mode)
                    
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit with cleanup."""
        self.cleanup()