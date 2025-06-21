# video.py
import json
import logging
import subprocess
import sys
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

    def __init__(self, video_path: Path, timestamps: np.ndarray, analytics_lookup: Optional[Dict] = None, 
                 use_sequential: bool = False):
        self.video_path = video_path
        self.timestamps = timestamps
        self.analytics_lookup = analytics_lookup or {}
        self.use_sequential = use_sequential
        self._cap: Optional[cv2.VideoCapture] = None
        self._current_idx = 0
        self._fps: Optional[float] = None
        
        # Sequential processing state
        self._current_frame_number = 0
        self._next_target_idx = 0

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

        if self.use_sequential:
            return self._next_sequential()
        else:
            return self._next_seeking()
    
    def _next_seeking(self) -> tuple[FrameInfo, np.ndarray]:
        """Get next frame using seek-based approach (original method)."""
        timestamp = self.timestamps[self._current_idx]
        frame_number = int(timestamp * self._fps)

        # Seek to frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self._cap.read()

        if not ret:
            raise StopIteration

        frame_info = self._create_frame_info(timestamp, frame_number)
        self._current_idx += 1
        return frame_info, frame
    
    def _next_sequential(self) -> tuple[FrameInfo, np.ndarray]:
        """Get next frame using sequential reading approach."""
        # Find next target frame number
        if self._next_target_idx >= len(self.timestamps):
            raise StopIteration
            
        target_timestamp = self.timestamps[self._next_target_idx]
        target_frame_number = int(target_timestamp * self._fps)
        
        # If we've already read past this frame (happens with duplicate frame numbers),
        # seek back to the target frame
        if self._current_frame_number > target_frame_number:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_number)
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration
            frame_info = self._create_frame_info(target_timestamp, target_frame_number)
            self._current_idx = self._next_target_idx
            self._next_target_idx += 1
            self._current_frame_number = target_frame_number + 1
            return frame_info, frame
        
        # Read frames sequentially until we reach the target
        while self._current_frame_number <= target_frame_number:
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration
                
            # If this is our target frame, return it
            if self._current_frame_number == target_frame_number:
                frame_info = self._create_frame_info(target_timestamp, target_frame_number)
                self._current_idx = self._next_target_idx
                self._next_target_idx += 1
                self._current_frame_number += 1
                return frame_info, frame
                
            self._current_frame_number += 1
            # Frame is automatically discarded here - no memory accumulation
            
        # Should never reach here if logic is correct
        raise StopIteration
    
    def _create_frame_info(self, timestamp: float, frame_number: int) -> FrameInfo:
        """Create FrameInfo with analytics lookup."""
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

        return frame_info


class FrameProcessor:
    """Handles frame processing and analytics alignment without storing frames in memory."""

    def __init__(self, video_path: Path, analytics_data: Optional[AnalyticsData] = None):
        self.video_path = Path(video_path)
        self.metadata = VideoMetadata(video_path)
        self.analytics_data = analytics_data

    def _should_use_sequential(self, sample_fps: float, video_fps: float, duration: float) -> bool:
        """Decide whether to use sequential or seeking approach based on efficiency."""
        # Calculate how many frames we'd need to read vs how many we want
        total_frames = int(duration * video_fps)
        target_frames = int(duration * sample_fps)
        
        # Use sequential if we're sampling more than 10% of frames
        # This threshold balances seek overhead vs extra frame reads
        sampling_ratio = target_frames / total_frames if total_frames > 0 else 0
        
        # Also consider absolute frame gaps - if gap is small, sequential is better
        avg_frame_gap = total_frames / target_frames if target_frames > 0 else float('inf')
        
        return sampling_ratio > 0.1 or avg_frame_gap < 10
    
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
        
        # Decide between sequential vs seeking approach
        use_sequential = self._should_use_sequential(fps, metadata['video']['fps'], duration)
        approach = "sequential" if use_sequential else "seeking"
        sampling_ratio = len(timestamps) / (duration * metadata['video']['fps']) * 100
        logger.info(f"Using {approach} extraction for {len(timestamps)} frames "
                   f"({sampling_ratio:.1f}% of video, sample_rate={fps:.2f} fps)")

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
        total_frames = len(timestamps)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting {total_frames} frames using {approach} approach...")
        
        with FrameIterator(self.video_path, timestamps, analytics_lookup, use_sequential) as frame_iter:
            for i, (frame_info, frame) in enumerate(frame_iter):
                # Update progress bar
                self._update_progress(i + 1, total_frames)
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
        
        # Clear progress bar and print completion
        print("\nFrame extraction completed!")

        # Create DataFrame with frame metadata
        df = pd.DataFrame(frame_records)

        # Convert timestamps if present - Fixed conversion logic
        if 'system_time' in df.columns:
            # Handle both string and numeric system_time values
            df['system_time'] = pd.to_datetime(pd.to_numeric(df['system_time'], errors='coerce'), unit='ms')

        return df
    
    def _update_progress(self, current: int, total: int, bar_length: int = 40):
        """Display ASCII progress bar."""
        percent = current / total
        filled_length = int(bar_length * percent)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Print progress bar with carriage return to overwrite
        sys.stdout.write(f'\r[{bar}] {current}/{total} ({percent:.1%})')
        sys.stdout.flush()