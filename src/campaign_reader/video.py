# video.py
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Union

import cv2
import numpy as np
import pandas as pd

from campaign_reader.analytics import AnalyticsData

logger = logging.getLogger(__name__)


class VideoMetadata:
    """Handles video metadata extraction and analysis."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._metadata: Optional[Dict] = None

    def extract_metadata(self) -> Dict:
        """Extract metadata using ffprobe."""
        if self._metadata is not None:
            return self._metadata

        try:
            # First check if file exists and has content
            if not self.video_path.exists() or self.video_path.stat().st_size == 0:
                raise ValueError(f"Video file is empty or doesn't exist: {self.video_path}")

            # Construct ffprobe command
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(self.video_path)
            ]

            # Run ffprobe
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)

            if not probe_data.get('streams'):
                raise ValueError("No streams found in video file")

            try:
                video_stream = next(
                    s for s in probe_data['streams']
                    if s['codec_type'] == 'video'
                )
            except StopIteration:
                raise ValueError("No video stream found in file")

            # Try to extract all fields with fallbacks
            self._metadata = {
                'duration': float(probe_data['format'].get('duration', 0)),
                'size_bytes': int(probe_data['format'].get('size', 0)),
                'format': probe_data['format'].get('format_name', 'unknown'),
                'video': {
                    'codec': video_stream.get('codec_name', 'unknown'),
                    'width': int(video_stream.get('width', 0)),
                    'height': int(video_stream.get('height', 0)),
                    'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                    'bitrate': int(video_stream.get('bit_rate', 0))
                }
            }

            return self._metadata

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to extract video metadata: {getattr(e, 'stderr', str(e))}")
        except Exception as e:
            raise ValueError(f"Unexpected error extracting video metadata: {str(e)}")


class Frame:
    """Represents a single video frame with its metadata."""

    def __init__(self, image: np.ndarray, timestamp: float, system_time: Optional[int] = None):
        self.image = image
        self.timestamp = timestamp  # Video timestamp in seconds
        self.system_time = system_time  # System timestamp from analytics
        self._analytics: Optional[Dict] = None

    def set_analytics(self, analytics: Dict) -> None:
        """Associate analytics data with this frame."""
        self._analytics = analytics

    @property
    def analytics(self) -> Optional[Dict]:
        """Get associated analytics data."""
        return self._analytics

    def save(self, path: Union[str, Path], format: str = 'jpg') -> None:
        """Save frame to disk."""
        path = Path(path)
        cv2.imwrite(str(path.with_suffix(f'.{format}')), self.image)


class FrameData:
    """Handles aligned frame and analytics data."""

    def __init__(self, frames: List[Frame], analytics_data: AnalyticsData):
        self.frames = frames
        self.analytics_data = analytics_data
        self._aligned_df: Optional[pd.DataFrame] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert aligned frame and analytics data to a DataFrame."""
        if self._aligned_df is not None:
            return self._aligned_df

        # Create frame data records
        frame_records = []
        for frame in self.frames:
            record = {
                'timestamp': frame.timestamp,
                'system_time': frame.system_time,
                'frame': frame.image
            }
            if frame.analytics:
                record.update(frame.analytics)
            frame_records.append(record)

        # Create DataFrame
        df = pd.DataFrame(frame_records)

        # Align with complete analytics data if any frames are missing analytics
        if self.analytics_data and df['system_time'].notna().any():
            analytics_df = self.analytics_data.to_dataframe()

            # Convert system_time to datetime if it isn't already
            if not pd.api.types.is_datetime64_any_dtype(df['system_time']):
                df['system_time'] = pd.to_datetime(df['system_time'].astype(float), unit='ms')

            # Merge frame data with analytics data
            df = pd.merge_asof(
                df.sort_values('system_time'),
                analytics_df.sort_values('systemTime'),
                left_on='system_time',
                right_on='systemTime',
                direction='nearest',
                tolerance=pd.Timedelta('100ms')  # Adjust tolerance as needed
            )

        self._aligned_df = df
        return df

    def save_frames(self, output_dir: Path, format: str = 'jpg') -> None:
        """Save all frames to the specified directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(self.frames):
            frame.save(output_dir / f"frame_{i:04d}", format=format)


class VideoFrameExtractor:
    """Handles video frame extraction and analytics alignment."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.metadata = VideoMetadata(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = None

    # ... [Previous VideoFrameExtractor methods remain unchanged]

    def extract_aligned_frames(self,
                               analytics_data: AnalyticsData,
                               sample_rate: Optional[float] = None) -> FrameData:
        """
        Extract frames and align them with analytics data.

        Args:
            analytics_data: Analytics data to align with frames
            sample_rate: Optional frames per second to extract (default: video fps)

        Returns:
            FrameData object containing aligned frames and analytics
        """
        self.open()

        # Get video metadata
        metadata = self.metadata.extract_metadata()
        video_duration = float(metadata['duration'])

        # Determine frame extraction rate
        fps = sample_rate or self._fps

        # Generate timestamps for frame extraction
        timestamps = np.arange(0, video_duration, 1 / fps)

        # Convert analytics data to DataFrame for timestamp extraction
        analytics_df = analytics_data.to_dataframe()
        video_times = analytics_df['videoTime'].dt.total_seconds().values

        # Extract frames
        frames = list(self.extract_frames(timestamps, analytics_df.to_dict('records')))

        return FrameData(frames, analytics_data)