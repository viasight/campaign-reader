# tests/test_video.py
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import os

import numpy as np
import pytest
import pandas as pd

from campaign_reader.analytics import AnalyticsData
from campaign_reader.models import CampaignSegment
from campaign_reader.video import VideoMetadata, FrameProcessor, FrameInfo, FrameIterator


@pytest.fixture
def sample_analytics(tmp_path):
    # Create sample analytics data
    analytics_data = [
        {
            'index': 0,
            'systemTime': '1000',
            'videoTime': '1000000',  # 1ms in ns
            'gps': {'latitude': 0.0, 'longitude': 0.0, 'accuracy': 1.0},
            'imu': {
                'linear_acceleration': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}
            }
        }
    ]

    # Write to temporary file
    analytics_file = tmp_path / "analytics.json"
    with open(analytics_file, 'w') as f:
        json.dump(analytics_data, f)

    return AnalyticsData([analytics_file])


@pytest.fixture
def mock_video_file(tmp_path):
    """Create a mock video file path."""
    video_path = tmp_path / "test.mp4"
    video_path.touch()
    return video_path


@pytest.fixture
def mock_metadata():
    """Mock video metadata."""
    return {
        'duration': 10.0,
        'size_bytes': 1000,
        'format': 'mp4',
        'video': {
            'codec': 'h264',
            'width': 1920,
            'height': 1080,
            'fps': 30.0,
            'bitrate': 1000000
        }
    }


def test_frame_iterator(mock_video_file, mock_metadata):
    """Test the FrameIterator."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Setup mocks
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap_instance.get.return_value = 30.0  # FPS
        mock_cap.return_value = mock_cap_instance

        # Test basic iteration
        timestamps = np.array([0.0, 1.0, 2.0])
        analytics_lookup = {
            0.0: {'systemTime': '1000', 'data': 'test1'},
            1.0: {'systemTime': '2000', 'data': 'test2'}
        }

        with FrameIterator(mock_video_file, timestamps, analytics_lookup) as frame_iter:
            frames = list(frame_iter)

            assert len(frames) == 3
            assert all(isinstance(frame_info, FrameInfo) for frame_info, _ in frames)
            assert all(isinstance(frame, np.ndarray) for _, frame in frames)

            # Check analytics alignment
            # Analytics should match exactly for 0.0 and 1.0
            assert frames[0][0].analytics['data'] == 'test1'
            assert frames[1][0].analytics['data'] == 'test2'
            # For 2.0, it should use nearest analytics within 1-second window (from 1.0)
            assert frames[2][0].analytics['data'] == 'test2'


def test_frame_processor_basic(mock_video_file, mock_metadata, tmp_path):
    """Test basic FrameProcessor functionality."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Setup mocks
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap_instance.get.return_value = 30.0  # FPS
        mock_cap.return_value = mock_cap_instance

        # Create mock metadata
        mock_metadata_instance = MagicMock(spec=VideoMetadata)
        mock_metadata_instance.extract_metadata.return_value = mock_metadata

        # Create processor
        processor = FrameProcessor(mock_video_file)
        processor.metadata = mock_metadata_instance

        # Test basic frame processing without saving
        df = processor.process_frames(sample_rate=1.0)

        assert isinstance(df, pd.DataFrame)
        assert 'timestamp' in df.columns
        assert 'frame_number' in df.columns
        assert 'frame_path' in df.columns
        assert df['frame_path'].isna().all()  # No paths when not saving

        # Test frame processing with saving
        output_dir = tmp_path / "frames"
        df = processor.process_frames(sample_rate=1.0, output_dir=output_dir)

        assert not df['frame_path'].isna().any()  # All frames should have paths
        assert all(os.path.dirname(path) == str(output_dir) for path in df['frame_path'])


def test_frame_processor_with_analytics(mock_video_file, mock_metadata, sample_analytics, tmp_path):
    """Test FrameProcessor with analytics data."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Setup mocks
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap_instance.get.return_value = 30.0  # FPS
        mock_cap.return_value = mock_cap_instance

        # Create mock metadata
        mock_metadata_instance = MagicMock(spec=VideoMetadata)
        mock_metadata_instance.extract_metadata.return_value = mock_metadata

        # Mock stat for video file
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=1024)  # Non-zero file size

            # Create processor with analytics
            processor = FrameProcessor(mock_video_file, sample_analytics)
            processor.metadata = mock_metadata_instance

            # Test processing with analytics
            df = processor.process_frames(sample_rate=1.0)

            assert isinstance(df, pd.DataFrame)
            assert 'timestamp' in df.columns
            assert 'system_time' in df.columns
            assert 'latitude' in df.columns
            assert 'longitude' in df.columns
            assert 'linear_acceleration_x' in df.columns


def test_frame_processor_custom_callback(mock_video_file, mock_metadata):
    """Test FrameProcessor with custom callback."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Setup mocks
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap_instance.get.return_value = 30.0  # FPS
        mock_cap.return_value = mock_cap_instance

        # Create mock metadata
        mock_metadata_instance = MagicMock(spec=VideoMetadata)
        mock_metadata_instance.extract_metadata.return_value = mock_metadata

        # Create processor
        processor = FrameProcessor(mock_video_file)
        processor.metadata = mock_metadata_instance

        # Test callback
        processed_frames = []

        def callback(frame_info, frame):
            processed_frames.append((frame_info, frame))

        processor.process_frames(sample_rate=1.0, frame_callback=callback)

        assert len(processed_frames) > 0
        assert all(isinstance(info, FrameInfo) for info, _ in processed_frames)
        assert all(isinstance(frame, np.ndarray) for _, frame in processed_frames)


def test_segment_frame_processing(tmp_path, mocker):
    """Test frame processing in CampaignSegment."""
    # Create a mock video file
    video_path = tmp_path / "segments" / "test-segment" / "video.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    # Set up analytics path first
    analytics_path = video_path.parent / "analytics" / "analytics.json"
    analytics_path.parent.mkdir(parents=True)

    # Set up all mocks at the top level
    with patch('pathlib.Path.stat') as mock_stat, \
            patch('pathlib.Path.exists') as mock_exists, \
            patch('pathlib.Path.glob') as mock_glob:
        mock_stat.return_value = MagicMock(st_size=1024)
        mock_exists.return_value = True
        mock_glob.return_value = [analytics_path]

        # Create analytics file
        with open(analytics_path, 'w') as f:
            json.dump([{
                'index': 0,
                'systemTime': 1000,  # Changed to numeric
                'videoTime': '1000000',
                'gps': {'latitude': 0.0, 'longitude': 0.0, 'accuracy': 1.0},
                'imu': {
                    'linear_acceleration': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'angular_velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0}
                }
            }], f)

        # Create test segment
        segment = CampaignSegment(
            id="test-segment",
            sequence_number=1,
            recorded_at=datetime.now(),
            video_path=str(video_path),
            analytics_file_pattern="analytics*.json"
        )
        segment._extracted_path = video_path.parent

        # Mock video metadata
        mock_metadata = {
            'duration': 10.0,
            'size_bytes': 1024,
            'format': 'mp4',
            'video': {
                'codec': 'h264',
                'width': 1920,
                'height': 1080,
                'fps': 30.0,
                'bitrate': 1000000
            }
        }

        # Test frame processing
        with patch('cv2.VideoCapture') as mock_cap, \
                patch('campaign_reader.video.VideoMetadata.extract_metadata', return_value=mock_metadata):
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
            mock_cap_instance.get.return_value = 30.0
            mock_cap.return_value = mock_cap_instance

            # Test basic processing
            df = segment.process_frames(sample_rate=1.0)
            assert isinstance(df, pd.DataFrame)
            assert not df.empty

            # Test processing with output directory
            output_dir = tmp_path / "output"
            df = segment.process_frames(sample_rate=1.0, output_dir=output_dir)
            assert not df['frame_path'].isna().any()
            assert all(Path(path).parent.name == "test-segment" for path in df['frame_path'])

            # Test processing without analytics
            df = segment.process_frames(sample_rate=1.0, align_analytics=False)
            assert 'latitude' not in df.columns

        # Test processing with missing video
        segment._extracted_path = None
        df = segment.process_frames()
        assert df is None


def test_video_metadata_extraction(mock_video_file, mock_metadata):
    """Test VideoMetadata extraction."""
    with patch('subprocess.run') as mock_run, \
            patch('pathlib.Path.stat') as mock_stat, \
            patch('pathlib.Path.exists') as mock_exists:
        # Mock file existence and size
        mock_exists.return_value = True
        mock_stat.return_value = MagicMock(st_size=1024)

        # Setup mock subprocess response
        mock_run.return_value = MagicMock(
            stdout=json.dumps({
                'format': {
                    'duration': '10.0',
                    'size': '1000',
                    'format_name': 'mp4'
                },
                'streams': [{
                    'codec_type': 'video',
                    'codec_name': 'h264',
                    'width': 1920,
                    'height': 1080,
                    'r_frame_rate': '30/1',
                    'bit_rate': '1000000'
                }]
            })
        )

        metadata = VideoMetadata(mock_video_file)
        result = metadata.extract_metadata()

        assert result['duration'] == 10.0
        assert result['video']['width'] == 1920
        assert result['video']['height'] == 1080
        assert result['video']['fps'] == 30.0
