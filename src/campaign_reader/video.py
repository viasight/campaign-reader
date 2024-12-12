# video.py
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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


@dataclass
class FrameInfo:
    """Lightweight class to store frame metadata without the actual image data."""
    timestamp: float
    system_time: Optional[int] = None
    analytics: Optional[Dict] = None
    frame_number: Optional[int] = None


class FrameIterator:
    """Iterator class that yields frames one at a time without storing them in memory."""

    def __init__(self, video_path: Path, timestamps: np.ndarray, analytics_lookup: Optional[Dict] = None):
        self.video_path = video_path
        self.timestamps = timestamps
        self.analytics_lookup = analytics_lookup or {}
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_idx = 0
        self._fps: Optional[float] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> None:
        """Open the video capture."""
        if self._cap is not None:
            return

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not self._fps:
            raise ValueError("Could not determine video FPS")

    def close(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self):
        return self

    def __next__(self) -> tuple[FrameInfo, np.ndarray]:
        """Get next frame and its metadata."""
        if self._current_idx >= len(self.timestamps):
            raise StopIteration

        if self._cap is None:
            raise ValueError("Video file not opened")

        timestamp = self.timestamps[self._current_idx]
        frame_number = int(timestamp * self._fps)

        # Seek to frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()

        if not ret:
            raise StopIteration

        # Create frame info
        frame_info = FrameInfo(
            timestamp=timestamp,
            frame_number=frame_number
        )

        # Find and attach nearest analytics record
        if self.analytics_lookup:
            # Use a wider matching window since we're matching based on system time
            window = 1.0  # 1 second window
            nearby_times = [t for t in self.analytics_lookup.keys()
                            if abs(t - timestamp) <= window]

            if nearby_times:
                nearest_time = min(nearby_times, key=lambda x: abs(x - timestamp))
                record = self.analytics_lookup[nearest_time]
                frame_info.analytics = record
                system_time = record.get('systemTime')
                if isinstance(system_time, str):
                    system_time = int(system_time)
                frame_info.system_time = system_time

        self._current_idx += 1
        return frame_info, frame


class FrameProcessor:
    """Handles frame processing and analytics alignment without storing frames in memory."""

    def __init__(self, video_path: Path, analytics_data: Optional[AnalyticsData] = None):
        self.video_path = Path(video_path)
        self.metadata = VideoMetadata(video_path)
        self.analytics_data = analytics_data

    def process_frames(self,
                       sample_rate: Optional[float] = None,
                       output_dir: Optional[Path] = None,
                       frame_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Process frames with optional saving and custom processing.

        Args:
            sample_rate: Frames per second to extract
            output_dir: Optional directory to save frames
            frame_callback: Optional callback function(frame_info, frame_data) for custom processing

        Returns:
            DataFrame with frame metadata, analytics, and frame_path (if output_dir specified)
        """
        # Get video metadata and sampling rate
        metadata = self.metadata.extract_metadata()
        fps = sample_rate or metadata['video']['fps']
        duration = float(metadata['duration'])

        # Generate timestamps
        timestamps = np.arange(0, duration, 1 / fps)

        # Create analytics lookup if available:
        analytics_lookup = {}
        if self.analytics_data:
            analytics_df = self.analytics_data.to_dataframe()

            # Use systemTime for matching instead of videoTime
            duration = float(metadata['duration'])

            # Normalize systemTime to be relative to start
            start_time = analytics_df['systemTime'].min()
            analytics_df['video_seconds'] = analytics_df['systemTime'] - start_time

            print(f"Analytics timing info:")
            print(f"System time range: {start_time} to {analytics_df['systemTime'].max()}")
            print(f"Duration from system time: {analytics_df['video_seconds'].max():.2f}s")
            print(f"Video duration: {duration:.2f}s")

            # Create lookup using normalized system time
            for _, row in analytics_df.iterrows():
                video_time = row['video_seconds']
                analytics_lookup[video_time] = row.to_dict()

            # Generate video frame timestamps
            fps = sample_rate or metadata['video']['fps']
            timestamps = np.arange(0, duration, 1 / fps)

        # Process frames
        frame_records = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        with FrameIterator(self.video_path, timestamps, analytics_lookup) as frame_iter:
            for frame_info, frame in frame_iter:
                # Save frame if output directory specified
                frame_path = None
                if output_dir:
                    frame_path = output_dir / f"frame_{frame_info.frame_number:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)

                # Call custom processing callback if provided
                if frame_callback:
                    frame_callback(frame_info, frame)

                # Add frame record without the image data
                record = {
                    'timestamp': frame_info.timestamp,
                    'system_time': frame_info.system_time,
                    'frame_number': frame_info.frame_number,
                    'frame_path': str(frame_path) if frame_path else None
                }

                if frame_info.analytics:
                    record.update(frame_info.analytics)

                frame_records.append(record)

        # Create DataFrame with frame metadata
        df = pd.DataFrame(frame_records)

        # Convert timestamps if present - Fixed conversion logic
        if 'system_time' in df.columns:
            # Handle both string and numeric system_time values
            df['system_time'] = pd.to_datetime(pd.to_numeric(df['system_time'], errors='coerce'), unit='ms')

        return df