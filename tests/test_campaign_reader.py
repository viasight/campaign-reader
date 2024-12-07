import json
import os
import tempfile
import zipfile
from datetime import datetime

import pytest

from campaign_reader import CampaignReader, CampaignZipError


@pytest.fixture
def sample_campaign_data():
    """Sample campaign metadata."""
    return {
        "id": "060953aa-7baf-435b-aee2-2faaeb438aaf",
        "name": "Test drive around Lees Summit",
        "createdAt": 1733434689359,
        "description": "Driving around Lees Summit MO",
        "segments": [
            {
                "id": "7e0226fd-2903-4d4b-b399-f103e9063e06",
                "sequenceNumber": 0,
                "recordedAt": 1733434791634,
                "videoPath": "/data/video.mp4",
                "analyticsFilePattern": "/data/analytics_{}"
            },
            {
                "id": "11eb4c5e-cc52-4836-b5a6-f68f64c0e1b9",
                "sequenceNumber": 1,
                "recordedAt": 1733435091748,
                "videoPath": "/data/video.mp4",
                "analyticsFilePattern": "/data/analytics_{}"
            }
        ]
    }

@pytest.fixture
def sample_analytics_data():
    """Sample analytics data."""
    return [
        {"timestamp": 1733434791634, "data": "sample1"},
        {"timestamp": 1733434791635, "data": "sample2"}
    ]

@pytest.fixture
def campaign_zip(sample_campaign_data, sample_analytics_data):
    """Create a temporary campaign zip file with test content."""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip.name, 'w') as zf:
            # Add campaign metadata
            zf.writestr('metadata/campaign.json', json.dumps(sample_campaign_data))
            
            # Add segment files
            for segment in sample_campaign_data['segments']:
                # Add video file
                segment_path = f"segments/{segment['id']}/video.mp4"
                zf.writestr(segment_path, b'fake video data')
                
                # Add analytics files
                analytics_path = f"segments/{segment['id']}/analytics/analytics.json"
                zf.writestr(analytics_path, json.dumps(sample_analytics_data))
                
                # Add additional analytics files
                for i in range(1, 3):
                    analytics_path = f"segments/{segment['id']}/analytics/analytics_{i}.json"
                    zf.writestr(analytics_path, json.dumps(sample_analytics_data))
        
        yield temp_zip.name
        # Cleanup
        os.unlink(temp_zip.name)

def test_campaign_metadata_loading(campaign_zip):
    """Test loading campaign metadata."""
    with CampaignReader(campaign_zip) as reader:
        campaign = reader.get_campaign_metadata()
        assert campaign.id == "060953aa-7baf-435b-aee2-2faaeb438aaf"
        assert campaign.name == "Test drive around Lees Summit"
        assert isinstance(campaign.created_at, datetime)
        assert campaign.description == "Driving around Lees Summit MO"
        assert len(campaign.segments) == 2

def test_segment_ordering(campaign_zip):
    """Test segment ordering."""
    with CampaignReader(campaign_zip) as reader:
        segments = reader.get_segments()
        assert len(segments) == 2
        assert segments[0].sequence_number == 0
        assert segments[1].sequence_number == 1
        assert segments[0].id == "7e0226fd-2903-4d4b-b399-f103e9063e06"

def test_segment_access(campaign_zip):
    """Test accessing individual segments."""
    with CampaignReader(campaign_zip) as reader:
        segment = reader.get_segment("7e0226fd-2903-4d4b-b399-f103e9063e06")
        assert segment is not None
        assert segment.sequence_number == 0
        assert isinstance(segment.recorded_at, datetime)
        
        # Test non-existent segment
        assert reader.get_segment("non-existent") is None

def test_segment_files_access(campaign_zip):
    """Test accessing segment files."""
    with CampaignReader(campaign_zip) as reader:
        segment = reader.get_segment("7e0226fd-2903-4d4b-b399-f103e9063e06")
        
        # Check video path
        video_path = segment.get_video_path()
        assert video_path is not None
        assert video_path.exists()
        assert video_path.name == "video.mp4"
        
        # Check analytics path
        analytics_path = segment.get_analytics_path()
        assert analytics_path is not None
        assert analytics_path.exists()
        assert analytics_path.is_dir()

def test_analytics_loading(campaign_zip):
    """Test loading analytics data."""
    with CampaignReader(campaign_zip) as reader:
        segment_id = "7e0226fd-2903-4d4b-b399-f103e9063e06"
        analytics = reader.get_segment_analytics(segment_id)
        
        # We have 3 files with 2 entries each
        assert len(analytics) == 6
        assert all(isinstance(entry["timestamp"], int) for entry in analytics)

def test_invalid_segment_analytics(campaign_zip):
    """Test error handling for invalid segment analytics."""
    with CampaignReader(campaign_zip) as reader:
        with pytest.raises(CampaignZipError):
            reader.get_segment_analytics("non-existent")

def test_segment_iteration(campaign_zip):
    """Test iterating through segments."""
    with CampaignReader(campaign_zip) as reader:
        segments = list(reader.iter_segments())
        assert len(segments) == 2
        assert [seg.sequence_number for seg in segments] == [0, 1]

def test_cleanup(campaign_zip):
    """Test cleanup of extracted files."""
    extracted_paths = []
    with CampaignReader(campaign_zip) as reader:
        # Store paths before cleanup
        segment = reader.get_segment("7e0226fd-2903-4d4b-b399-f103e9063e06")
        video_path = segment.get_video_path()
        analytics_path = segment.get_analytics_path()
        extracted_paths.extend([video_path, analytics_path])
        
        # Verify files exist
        assert all(path.exists() for path in extracted_paths if path is not None)
    
    # After context manager exits, verify cleanup
    assert not any(path.exists() for path in extracted_paths if path is not None)