# tests/test_video.py
import json
from datetime import datetime

import numpy as np
import pytest

from campaign_reader.analytics import AnalyticsData
from campaign_reader.models import CampaignSegment
from campaign_reader.video import Frame, FrameData


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