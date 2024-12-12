from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Union, Callable
from pathlib import Path
import pandas as pd

from campaign_reader.analytics import AnalyticsData
from campaign_reader.video import VideoMetadata, FrameProcessor, logger


@dataclass
class CampaignSegment:
    """Represents a segment in a campaign."""
    id: str
    sequence_number: int
    recorded_at: datetime
    video_path: str
    analytics_file_pattern: str
    _extracted_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'CampaignSegment':
        """Create a segment from a dictionary."""
        return cls(
            id=data['id'],
            sequence_number=data['sequenceNumber'],
            recorded_at=datetime.fromtimestamp(data['recordedAt'] / 1000.0),
            video_path=data['videoPath'],
            analytics_file_pattern=data['analyticsFilePattern']
        )

    def get_video_path(self) -> Optional[Path]:
        """Get the path to the extracted video file."""
        if self._extracted_path:
            return self._extracted_path / 'video.mp4'
        return None

    def get_analytics_path(self) -> Optional[Path]:
        """Get the path to the analytics directory."""
        if self._extracted_path:
            return self._extracted_path / 'analytics'
        return None

    def get_analytics_files(self) -> List[Path]:
        """Get all analytics files for this segment in order."""
        if not self._extracted_path:
            return []

        analytics_path = self.get_analytics_path()
        logger.info(f"analytics_path: {analytics_path}")
        if not analytics_path or not analytics_path.exists():
            return []

        paths = analytics_path.glob('analytics*.json')
        if not analytics_path.exists():
            return []
        sorted_paths = sorted(paths)

        return sorted_paths

    def get_analytics_data(self) -> Optional[AnalyticsData]:
        """Get analytics data handler for this segment."""
        files = self.get_analytics_files()
        logger.info(f"files: {files}")
        return AnalyticsData(files) if files else None

    def get_video_metadata(self) -> Optional[Dict]:
        """Get video metadata for this segment."""
        video_path = self.get_video_path()
        if not video_path or not video_path.exists():
            return None

        return VideoMetadata(video_path).extract_metadata()

    def process_frames(self,
                       sample_rate: Optional[float] = None,
                       output_dir: Optional[Union[str, Path]] = None,
                       frame_callback: Optional[Callable] = None,
                       align_analytics: bool = True) -> Optional[pd.DataFrame]:
        """
        Process frames from the video with optional saving and custom processing.

        Args:
            sample_rate: Optional frames per second to extract (default: video fps)
            output_dir: Optional directory to save frames
            frame_callback: Optional callback function(frame_info, frame_data) for custom processing
            align_analytics: Whether to align frames with analytics data

        Returns:
            DataFrame containing frame metadata and analytics if successful,
            None otherwise
        """
        video_path = self.get_video_path()
        if not video_path or not video_path.exists():
            logger.error(f"Video file not found for segment {self.id}")
            return None

        try:
            # Get analytics data if required
            analytics_data = self.get_analytics_data() if align_analytics else None
            if align_analytics and not analytics_data:
                logger.error(f"Analytics data not found for segment {self.id}")
                return None

            # Create frame processor
            processor = FrameProcessor(video_path, analytics_data)

            # Process frames
            if output_dir:
                output_dir = Path(output_dir) / self.id

            return processor.process_frames(
                sample_rate=sample_rate,
                output_dir=output_dir,
                frame_callback=frame_callback
            )

        except Exception as e:
            logger.error(f"Failed to process frames from segment {self.id}: {str(e)}")
            return None

    def extract_frames(self,
                       sample_rate: Optional[float] = None,
                       align_analytics: bool = True,
                       output_dir: Optional[Union[str, Path]] = None) -> Optional[pd.DataFrame]:
        """
        Legacy method for backward compatibility. Extracts frames and saves them if output_dir is provided.

        Args:
            sample_rate: Optional frames per second to extract (default: video fps)
            align_analytics: Whether to align frames with analytics data
            output_dir: Optional directory to save frames

        Returns:
            DataFrame containing frame metadata and analytics
        """
        return self.process_frames(
            sample_rate=sample_rate,
            align_analytics=align_analytics,
            output_dir=output_dir
        )


@dataclass
class Campaign:
    """Represents a campaign with metadata and segments."""
    id: str
    name: str
    created_at: datetime
    description: str
    segments: List[CampaignSegment]

    @classmethod
    def from_dict(cls, data: dict) -> 'Campaign':
        """Create a campaign from a dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            created_at=datetime.fromtimestamp(data['createdAt'] / 1000.0),
            description=data['description'],
            segments=[CampaignSegment.from_dict(seg) for seg in data['segments']]
        )

    def get_segment(self, segment_id: str) -> Optional[CampaignSegment]:
        """Get a segment by its ID."""
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        return None

    def get_ordered_segments(self) -> List[CampaignSegment]:
        """Get segments ordered by sequence number."""
        return sorted(self.segments, key=lambda x: x.sequence_number)
