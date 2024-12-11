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

    def save(self, path: Union[str, Path], img_format: str = 'jpg') -> None:
        """Save frame to disk."""
        path = Path(path)
        cv2.imwrite(str(path.with_suffix(f'.{img_format}')), self.image)


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

        # Align with complete analytics data if any frames have analytics
        if self.analytics_data and not df['system_time'].isna().all():
            analytics_df = self.analytics_data.to_dataframe()

            # Convert system_time to datetime if it isn't already
            if not pd.api.types.is_datetime64_any_dtype(df['system_time']):
                df['system_time'] = pd.to_datetime(df['system_time'].astype(float), unit='ms')

            # Create a mask for valid system_time entries
            valid_mask = df['system_time'].notna()

            if valid_mask.any():
                # Split into valid and invalid frames
                valid_frames_df = df[valid_mask].copy()
                invalid_frames_df = df[~valid_mask].copy()

                # Merge valid frames with analytics
                merged_df = pd.merge_asof(
                    valid_frames_df.sort_values('system_time'),
                    analytics_df.sort_values('systemTime'),
                    left_on='system_time',
                    right_on='systemTime',
                    direction='nearest',
                    tolerance=pd.Timedelta('100ms')  # Adjust tolerance as needed
                )

                # Add missing columns to invalid_frames_df
                for col in merged_df.columns:
                    if col not in invalid_frames_df.columns:
                        invalid_frames_df[col] = None

                # Create a list to store frames in original order
                final_frames = []
                for i in range(len(df)):
                    if valid_mask.iloc[i]:
                        # Find corresponding row in merged_df
                        matched_row = merged_df[merged_df['system_time'] == df['system_time'].iloc[i]]
                        if not matched_row.empty:
                            final_frames.append(matched_row.iloc[0])
                        else:
                            # If no match found (shouldn't happen), use row with nulls
                            null_row = pd.Series({col: None for col in merged_df.columns})
                            null_row[['timestamp', 'system_time', 'frame']] = df.iloc[i][
                                ['timestamp', 'system_time', 'frame']]
                            final_frames.append(null_row)
                    else:
                        # Use invalid frame
                        final_frames.append(invalid_frames_df.iloc[(~valid_mask)[:i].sum()])

                # Combine all frames into final DataFrame
                df = pd.DataFrame(final_frames)

            else:
                # If no valid frames with system_time, add null columns for all analytics fields
                analytics_columns = set(analytics_df.columns) - set(df.columns)
                for col in analytics_columns:
                    df[col] = None

        self._aligned_df = df
        return df

    def save_frames(self, output_dir: Path, img_format: str = 'jpg') -> None:
        """Save all frames to the specified directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(self.frames):
            frame.save(output_dir / f"frame_{i:04d}", img_format=img_format)


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

        # Extract frames
        frames = list(self.extract_frames(timestamps, analytics_df.to_dict('records')))

        return FrameData(frames, analytics_data)

    def open(self) -> None:
        """
        Opens the video file and initializes the video capture object.

        Raises:
            ValueError: If the video file cannot be opened or is invalid
        """
        if self._cap is not None:
            return

        try:
            self._cap = cv2.VideoCapture(str(self.video_path))

            if not self._cap.isOpened():
                raise ValueError(f"Failed to open video file: {self.video_path}")

            # Get video FPS from metadata
            metadata = self.metadata.extract_metadata()
            self._fps = metadata['video']['fps']

            if not self._fps:
                # Fallback to getting FPS directly from the capture object
                self._fps = self._cap.get(cv2.CAP_PROP_FPS)

            if not self._fps:
                raise ValueError("Could not determine video FPS")

        except Exception as e:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            raise ValueError(f"Error opening video file: {str(e)}")

    def extract_frames(self, timestamps: np.ndarray, analytics: Optional[List[Dict]] = None) -> List[Frame]:
        """
        Extract frames at specific timestamps and optionally align with analytics data.

        Args:
            timestamps: Array of timestamps (in seconds) at which to extract frames
            analytics: Optional list of analytics records to align with frames

        Returns:
            List of Frame objects containing the extracted frames

        Raises:
            ValueError: If the video capture object is not initialized or frame extraction fails
        """
        if self._cap is None:
            raise ValueError("Video file not opened. Call open() first.")

        frames = []
        fps = self._fps
        frame_duration = 1.0 / fps

        # Create analytics lookup if provided
        analytics_lookup = {}
        if analytics:
            for record in analytics:
                # Convert video time to seconds
                if isinstance(record['videoTime'], pd.Timedelta):
                    video_time = record['videoTime'].total_seconds()
                else:
                    # Assuming nanosecond string
                    video_time = float(record['videoTime']) / 1e9
                analytics_lookup[video_time] = record

        try:
            for timestamp in timestamps:
                # Calculate frame number and seek to position
                frame_number = int(float(timestamp) * fps)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                # Read frame
                ret, frame_img = self._cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame at timestamp {timestamp:.2f}")
                    continue

                # Create Frame object
                frame = Frame(
                    image=frame_img,
                    timestamp=float(timestamp),
                    system_time=None  # Will be set from analytics if available
                )

                # Find and attach nearest analytics record
                if analytics_lookup:
                    # Look for analytics within half a frame duration
                    nearest_time = min(
                        analytics_lookup.keys(),
                        key=lambda x: abs(x - float(timestamp))
                    )
                    if abs(nearest_time - float(timestamp)) <= frame_duration / 2:
                        record = analytics_lookup[nearest_time]
                        frame.set_analytics(record)
                        # Convert system time to integer milliseconds if it's a string
                        system_time = record['systemTime']
                        if isinstance(system_time, str):
                            system_time = int(system_time)
                        frame.system_time = system_time

                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Error extracting frames: {str(e)}")

        return frames
