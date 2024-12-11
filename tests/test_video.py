# tests/test_video.py
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from campaign_reader.analytics import AnalyticsData
from campaign_reader.models import CampaignSegment
from campaign_reader.video import Frame, FrameData, VideoFrameExtractor, VideoMetadata


@pytest.fixture
def sample_frame():
    # Create a simple test image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    return Frame(image, timestamp=1.0, system_time=1000)


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

def test_frame_analytics_association(sample_frame):
    analytics = {'test': 'data'}
    sample_frame.set_analytics(analytics)
    assert sample_frame.analytics == analytics


def test_frame_data_creation(sample_frame, sample_analytics):
    frame_data = FrameData([sample_frame], sample_analytics)
    df = frame_data.to_dataframe()

    assert not df.empty
    assert 'frame' in df.columns
    assert 'timestamp' in df.columns
    assert 'system_time' in df.columns


def test_frame_data_alignment(sample_frame, sample_analytics):
    frame_data = FrameData([sample_frame], sample_analytics)
    df = frame_data.to_dataframe()

    # Check that GPS and IMU data is present
    assert 'latitude' in df.columns
    assert 'longitude' in df.columns
    assert 'linear_acceleration_x' in df.columns


def test_segment_frame_extraction(tmp_path, mocker):
    # Create a mock video file
    video_path = tmp_path / "segments" / "test-segment" / "video.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.touch()

    # Create a mock analytics file
    analytics_path = video_path.parent / "analytics" / "analytics.json"
    analytics_path.parent.mkdir(parents=True)
    analytics_path.touch()

    # Create test segment
    segment = CampaignSegment(
        id="test-segment",
        sequence_number=1,
        recorded_at=datetime.now(),
        video_path=str(video_path),
        analytics_file_pattern="analytics*.json"
    )
    segment._extracted_path = video_path.parent

    # Create a mock VideoFrameExtractor instance
    mock_frame_extractor = mocker.MagicMock()
    mock_frame_extractor.extract_aligned_frames.return_value = mocker.Mock(spec=FrameData)

    # Mock the VideoFrameExtractor class constructor
    mocker.patch('campaign_reader.models.VideoFrameExtractor', return_value=mock_frame_extractor)

    # Test frame extraction
    frame_data = segment.extract_frames(sample_rate=1.0)
    assert frame_data is not None
    mock_frame_extractor.extract_aligned_frames.assert_called_once()

    # Test frame extraction with missing video
    segment._extracted_path = None
    frame_data = segment.extract_frames()
    assert frame_data is None


def test_video_frame_extractor_open(mock_video_file, mock_metadata):
    """Test VideoFrameExtractor.open() method."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Setup mocks
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap.return_value = mock_cap_instance

        # Create a mock VideoMetadata instance
        mock_metadata_instance = MagicMock(spec=VideoMetadata)
        mock_metadata_instance.extract_metadata.return_value = mock_metadata

        # Create the extractor with the mock metadata
        extractor = VideoFrameExtractor(mock_video_file)
        extractor.metadata = mock_metadata_instance  # Replace the metadata instance

        # Test successful open
        extractor.open()

        assert extractor._cap is not None
        assert extractor._fps == 30.0
        mock_cap.assert_called_once_with(str(mock_video_file))

        # Test reopen doesn't create new capture
        extractor.open()
        mock_cap.assert_called_once()  # Should still only be called once

        # Test failed open
        mock_cap_instance.isOpened.return_value = False
        extractor = VideoFrameExtractor(mock_video_file)
        extractor.metadata = mock_metadata_instance  # Replace the metadata instance
        with pytest.raises(ValueError, match="Failed to open video file"):
            extractor.open()


def test_video_frame_extractor_extract_frames(mock_video_file, mock_metadata):
    """Test VideoFrameExtractor.extract_frames() method."""
    with patch('cv2.VideoCapture') as mock_cap:
        # Setup mocks
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap_instance.get.return_value = 30.0  # FPS
        mock_cap.return_value = mock_cap_instance

        # Create a mock VideoMetadata instance
        mock_metadata_instance = MagicMock(spec=VideoMetadata)
        mock_metadata_instance.extract_metadata.return_value = mock_metadata

        # Create extractor and test timestamps
        extractor = VideoFrameExtractor(mock_video_file)
        extractor.metadata = mock_metadata_instance  # Replace the metadata instance
        extractor.open()

        timestamps = np.array([0.0, 1.0, 2.0])
        analytics = [
            {
                'videoTime': '0',
                'systemTime': '1000',
                'data': 'test1'
            },
            {
                'videoTime': '1000000000',  # 1 second in ns
                'systemTime': '2000',
                'data': 'test2'
            }
        ]

        frames = extractor.extract_frames(timestamps, analytics)

        # Verify basic extraction
        assert len(frames) == 3
        assert all(isinstance(f, Frame) for f in frames)
        assert all(f.image.shape == (100, 100, 3) for f in frames)

        # Verify timestamps
        assert [f.timestamp for f in frames] == [0.0, 1.0, 2.0]

        # Verify analytics alignment
        assert frames[0].analytics is not None
        assert frames[0].analytics['data'] == 'test1'
        assert frames[1].analytics is not None
        assert frames[1].analytics['data'] == 'test2'
        assert frames[2].analytics is None  # No analytics for t=2.0

        # Test extraction without analytics
        frames = extractor.extract_frames(timestamps)
        assert len(frames) == 3
        assert all(f.analytics is None for f in frames)

        # Test failed frame read
        mock_cap_instance.read.return_value = (False, None)
        frames = extractor.extract_frames(timestamps)
        assert len(frames) == 0


def test_video_frame_extractor_error_handling(mock_video_file):
    """Test error handling in VideoFrameExtractor."""
    # Test extract_frames without opening
    extractor = VideoFrameExtractor(mock_video_file)
    with pytest.raises(ValueError, match="Video file not opened"):
        extractor.extract_frames(np.array([0.0]))

    # Test cleanup on error
    with patch('cv2.VideoCapture') as mock_cap:
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_cap.return_value = mock_cap_instance

        extractor = VideoFrameExtractor(mock_video_file)
        with pytest.raises(ValueError):
            extractor.open()

        assert extractor._cap is None
        mock_cap_instance.release.assert_called_once()