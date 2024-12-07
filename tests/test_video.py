# tests/test_video.py

import subprocess

import pytest

from campaign_reader import CampaignReader
from campaign_reader.video import VideoMetadata


@pytest.fixture
def sample_video_file(tmp_path):
    """Create a dummy MP4 file for testing."""
    video_path = tmp_path / 'test_video.mp4'

    # Create minimal valid MP4 file
    try:
        subprocess.run([
            'ffmpeg',
            '-f', 'lavfi',
            '-i', 'color=c=black:s=1280x720:r=30:d=1',
            '-c:v', 'libx264',
            '-t', '1',
            str(video_path)
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pytest.skip("ffmpeg not available - skipping video tests")

    return video_path


def test_video_metadata_extraction(sample_video_file):
    """Test basic video metadata extraction."""
    video = VideoMetadata(sample_video_file)
    try:
        metadata = video.extract_metadata()

        # Check basic metadata structure
        assert isinstance(metadata, dict)
        assert 'duration' in metadata
        assert 'size_bytes' in metadata
        assert 'video' in metadata

        # Check video-specific metadata
        video_meta = metadata['video']
        assert video_meta['width'] == 1280
        assert video_meta['height'] == 720
        assert video_meta['fps'] == 30
        assert video_meta['codec'] == 'h264'
    except ValueError as e:
        if 'ffmpeg not available' in str(e):
            pytest.skip("ffmpeg not available - skipping metadata extraction test")
        raise


def test_video_metadata_caching(sample_video_file):
    """Test that metadata is cached after first extraction."""
    video = VideoMetadata(sample_video_file)

    try:
        # First extraction
        metadata1 = video.extract_metadata()

        # Should use cached version
        metadata2 = video.extract_metadata()

        assert metadata1 is metadata2  # Should be the same object
    except ValueError as e:
        if 'ffmpeg not available' in str(e):
            pytest.skip("ffmpeg not available - skipping metadata cache test")
        raise


def test_video_metadata_invalid_file(tmp_path):
    """Test handling of invalid video files."""
    invalid_file = tmp_path / 'invalid.mp4'
    invalid_file.write_bytes(b'not a video file')

    video = VideoMetadata(invalid_file)
    with pytest.raises(ValueError) as exc_info:
        video.extract_metadata()

    assert 'Failed to extract video metadata' in str(exc_info.value)


def test_segment_analytics_df(campaign_zip):
    """Test getting analytics data as DataFrame."""
    with CampaignReader(campaign_zip) as reader:
        segment_id = reader.get_segments()[0].id
        df = reader.get_segment_analytics_df(segment_id)

        assert df is not None
        assert not df.empty
        assert 'systemTime' in df.columns
        assert 'latitude' in df.columns


def test_segment_video_metadata(campaign_zip):
    """Test getting video metadata for a segment."""
    with CampaignReader(campaign_zip) as reader:
        segment = reader.get_segments()[0]

        try:
            metadata = reader.get_segment_video_metadata(segment.id)
            # Basic checks only - full content tested in direct video tests
            assert isinstance(metadata, dict)
            assert 'duration' in metadata
            assert 'video' in metadata
        except ValueError as e:
            if 'ffmpeg not available' in str(e):
                pytest.skip("ffmpeg not available - skipping segment video metadata test")
            raise


def test_validate_segment_analytics(campaign_zip):
    """Test analytics validation for a segment."""
    with CampaignReader(campaign_zip) as reader:
        segment = reader.get_segments()[0]
        validation_results = reader.validate_segment_analytics(segment.id)

        assert isinstance(validation_results, dict)
        assert 'errors' in validation_results
        assert 'warnings' in validation_results
