# video.py
from typing import Dict, Optional, List, Union, Generator, Tuple
import subprocess
import json
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import logging

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


class VideoFrameExtractor:
    """Handles video frame extraction and analytics alignment."""

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.metadata = VideoMetadata(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> None:
        """Open the video file."""
        if self._cap is not None:
            return

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")
        
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = self.metadata.extract_metadata()['video']['fps']

    def close(self) -> None:
        """Close the video file."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame_at_time(self, timestamp: float) -> Optional[Frame]:
        """Extract a single frame at the specified timestamp (in seconds)."""
        if self._cap is None:
            self.open()

        # Calculate frame number from timestamp
        frame_number = int(timestamp * self._fps)
        
        # Set position
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = self._cap.read()
        if not ret:
            return None

        return Frame(frame, timestamp)

    def extract_frames(self, 
                      timestamps: List[float],
                      analytics_data: Optional[List[Dict]] = None) -> Generator[Frame, None, None]:
        """Extract multiple frames at specified timestamps and optionally align with analytics."""
        if self._cap is None:
            self.open()

        # Sort timestamps for efficient extraction
        timestamps = sorted(timestamps)
        
        # Create analytics lookup if provided
        analytics_lookup = {}
        if analytics_data:
            for entry in analytics_data:
                video_time = float(entry['videoTime']) / 1000  # Convert to seconds
                analytics_lookup[video_time] = entry

        # Extract frames
        for timestamp in timestamps:
            frame = self.get_frame_at_time(timestamp)
            if frame is None:
                logger.warning(f"Failed to extract frame at timestamp {timestamp}")
                continue

            # Align with analytics if available
            if analytics_lookup:
                # Find closest analytics entry
                closest_time = min(analytics_lookup.keys(),
                                 key=lambda x: abs(x - timestamp))
                if abs(closest_time - timestamp) < 1.0/self._fps:  # Within one frame
                    frame.set_analytics(analytics_lookup[closest_time])

            yield frame

    def batch_extract(self, 
                     start_time: float,
                     end_time: float,
                     interval: float,
                     analytics_data: Optional[List[Dict]] = None) -> Generator[Frame, None, None]:
        """Extract frames at regular intervals between start and end times."""
        timestamps = np.arange(start_time, end_time, interval)
        yield from self.extract_frames(timestamps.tolist(), analytics_data)

