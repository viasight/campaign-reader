import os
import json
import zipfile
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Generator
import logging
from .models import Campaign, CampaignSegment

import pandas as pd

logger = logging.getLogger(__name__)

class CampaignZipError(Exception):
    """Base exception for campaign zip file errors."""
    pass

class CampaignReader:
    """A class for reading and analyzing campaign zip files."""
    
    def __init__(self, zip_path: str, extract_dir: Optional[str] = None, require_campaign_metadata: bool = False):
        """Initialize the CampaignReader with a path to a zip file.
        
        Args:
            zip_path (str): Path to the campaign zip file
            extract_dir (Optional[str]): Directory to extract files to. If None, uses a temp directory
            require_campaign_metadata (bool): If True, validates campaign structure and loads metadata
            
        Raises:
            FileNotFoundError: If zip_path doesn't exist
            CampaignZipError: If zip file is invalid or extraction fails
        """
        self.zip_path = Path(zip_path)
        self.require_campaign_metadata = require_campaign_metadata
        self._campaign = None
        
        self._validate_zip_file()
        
        # Set up extraction directory
        self._using_temp_dir = extract_dir is None
        self._extract_dir = Path(extract_dir) if extract_dir else Path(tempfile.mkdtemp())
        self._extracted_files: Dict[str, Path] = {}
        
        # Extract contents
        self._extract_contents()
        
        # Load campaign if required
        if self.require_campaign_metadata:
            self._load_campaign()

    def _load_campaign(self) -> None:
        """Load campaign metadata from the extracted files."""
        try:
            metadata_path = self._extract_dir / 'metadata' / 'campaign.json'
            with open(metadata_path) as f:
                campaign_data = json.load(f)
            
            self._campaign = Campaign.from_dict(campaign_data)
            
            # Set extracted paths for segments
            for segment in self._campaign.segments:
                segment._extracted_path = self._extract_dir / 'segments' / segment.id
                
        except Exception as e:
            raise CampaignZipError(f"Failed to load campaign metadata: {str(e)}")

    def _validate_zip_file(self) -> None:
        """Validate that the zip file exists and can be opened."""
        if not self.zip_path.exists():
            raise FileNotFoundError(f"File {self.zip_path} not found")
            
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                if zf.testzip() is not None:
                    raise CampaignZipError(f"Zip file {self.zip_path} is corrupted")
                
                self.file_list = zf.namelist()
                
                # Check for required files only if campaign metadata is required
                if self.require_campaign_metadata and 'metadata/campaign.json' not in self.file_list:
                    raise CampaignZipError("Campaign metadata file not found")
                
                # Basic security check for zip slip
                for fname in self.file_list:
                    if os.path.isabs(fname) or fname.startswith('..'):
                        raise CampaignZipError(f"Potentially malicious path in zip: {fname}")
                        
        except zipfile.BadZipFile:
            raise CampaignZipError(f"File {self.zip_path} is not a valid zip file")

    def _extract_contents(self) -> None:
        """Extract the contents of the zip file to the extraction directory."""
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                self._extract_dir.mkdir(parents=True, exist_ok=True)
                
                for fname in self.file_list:
                    extract_path = self._extract_dir / fname
                    
                    if not extract_path.resolve().is_relative_to(self._extract_dir.resolve()):
                        raise CampaignZipError(f"Attempted path traversal: {fname}")
                    
                    extract_path.parent.mkdir(parents=True, exist_ok=True)
                    zf.extract(fname, self._extract_dir)
                    self._extracted_files[fname] = extract_path
                    
        except Exception as e:
            self.cleanup()
            raise CampaignZipError(f"Failed to extract zip contents: {str(e)}")
    
    def get_campaign_metadata(self) -> Campaign:
        """Get the campaign metadata."""
        if not self._campaign:
            if not self.require_campaign_metadata:
                self._load_campaign()
        return self._campaign
    
    def get_segment(self, segment_id: str) -> Optional[CampaignSegment]:
        """Get a specific segment by ID."""
        campaign = self.get_campaign_metadata()
        return campaign.get_segment(segment_id) if campaign else None
    
    def get_segments(self) -> List[CampaignSegment]:
        """Get all segments in sequence order."""
        campaign = self.get_campaign_metadata()
        return campaign.get_ordered_segments() if campaign else []
    
    def iter_segments(self) -> Generator[CampaignSegment, None, None]:
        """Iterate through segments in sequence order."""
        campaign = self.get_campaign_metadata()
        if campaign:
            for segment in campaign.get_ordered_segments():
                yield segment
    
    def get_segment_analytics(self, segment_id: str) -> List[dict]:
        """Get analytics data for a specific segment."""
        segment = self.get_segment(segment_id)
        if not segment:
            raise CampaignZipError(f"Segment {segment_id} not found")
            
        analytics_dir = segment.get_analytics_path()
        if not analytics_dir or not analytics_dir.exists():
            raise CampaignZipError(f"Analytics directory not found for segment {segment_id}")
            
        analytics_data = []
        try:
            for analytics_file in sorted(analytics_dir.glob('analytics*.json')):
                with open(analytics_file) as f:
                    analytics_data.extend(json.load(f))
        except Exception as e:
            raise CampaignZipError(f"Failed to load analytics data: {str(e)}")
            
        return analytics_data
    
    def get_extracted_file(self, filename: str) -> Optional[Path]:
        """Get path to an extracted file."""
        return self._extracted_files.get(filename)

    def get_segment_analytics_df(self, segment_id: str) -> pd.DataFrame:
        """Get analytics data as a pandas DataFrame for a segment."""
        segment = self.get_segment(segment_id)
        if not segment:
            raise CampaignZipError(f"Segment {segment_id} not found")

        analytics = segment.get_analytics_data()
        if not analytics:
            raise CampaignZipError(
                f"No analytics data found for segment {segment_id}"
            )

        return analytics.to_dataframe()

    def get_segment_video_metadata(self, segment_id: str) -> Dict:
        """Get video metadata for a segment."""
        segment = self.get_segment(segment_id)
        if not segment:
            raise CampaignZipError(f"Segment {segment_id} not found")

        metadata = segment.get_video_metadata()
        if not metadata:
            raise CampaignZipError(
                f"Failed to get video metadata for segment {segment_id}"
            )

        return metadata

    def validate_segment_analytics(self, segment_id: str) -> Dict[str, List[str]]:
        """Validate analytics data for a segment."""
        segment = self.get_segment(segment_id)
        if not segment:
            raise CampaignZipError(f"Segment {segment_id} not found")

        analytics = segment.get_analytics_data()
        if not analytics:
            raise CampaignZipError(
                f"No analytics data found for segment {segment_id}"
            )

        return analytics.validate()
    
    def list_files(self) -> List[str]:
        """Get a list of all files in the zip."""
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
                                pass
                    
                    try:
                        self._extract_dir.rmdir()
                    except OSError:
                        pass

                # Restore original permissions if we changed them
                if not os.access(self._extract_dir, os.W_OK):
                    os.chmod(self._extract_dir, current_mode)
                    
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()