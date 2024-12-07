import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import tempfile
import zipfile
from unittest.mock import Mock, patch

from campaign_reader.reader import CampaignReader, CampaignZipError
from campaign_reader.models import Campaign, CampaignSegment

@pytest.fixture
def sample_analytics_data():
    """Generate sample analytics data with both IMU and GPS readings."""
    base_time = 1733327149354
    return [
        {
            "index": i,
            "systemTime": str(base_time + i * 100),
            "videoTime": str(i * 33478000),
            "gps": {
                "latitude": 44.9553195 + (i * 0.0001),
                "longitude": -93.3773398 + (i * 0.0001),
                "accuracy": 14.813
            } if i % 3 != 0 else None,  # Simulate occasional missing GPS
            "imu": {
                "linear_acceleration": {
                    "x": -0.020338991656899452,
                    "y": 0.267397940158844,
                    "z": 9.778867721557617
                },
                "angular_velocity": {
                    "x": -0.001527163083665073,
                    "y": -0.0024434609804302454,
                    "z": 1.5271631127689034E-4
                }
            }
        } for i in range(20)
    ]

@pytest.fixture
def sample_campaign_zip(tmp_path, sample_analytics_data):
    """Create a sample campaign ZIP file with analytics data."""
    campaign_data = {
        "id": "test-campaign",
        "name": "Test Campaign",
        "createdAt": 1733327149354,
        "description": "Test campaign with GPS and IMU data",
        "segments": [{
            "id": "test-segment",
            "sequenceNumber": 0,
            "recordedAt": 1733327149354,
            "videoPath": "/data/video.mp4",
            "analyticsFilePattern": "/data/analytics_{}.json"
        }]
    }
    
    zip_path = tmp_path / "test_campaign.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add campaign metadata
        zf.writestr('metadata/campaign.json', json.dumps(campaign_data))
        
        # Add analytics data
        analytics_str = '\n'.join(json.dumps(d) for d in sample_analytics_data)
        zf.writestr('segments/test-segment/analytics/analytics.json', analytics_str)
        
        # Add dummy video file
        zf.writestr('segments/test-segment/video.mp4', b'dummy video data')
    
    return zip_path

def test_get_segment_analytics_df_basic(sample_campaign_zip):
    """Test basic DataFrame conversion with flattened columns."""
    reader = CampaignReader(sample_campaign_zip, require_campaign_metadata=True)
    df = reader.get_segment_analytics_df("test-segment")
    
    # Check basic DataFrame properties
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.index.name == 'systemTime'
    
    # Check that IMU data was flattened
    expected_imu_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    assert all(col in df.columns for col in expected_imu_columns)
    
    # Check that GPS data was flattened
    expected_gps_columns = ['latitude', 'longitude', 'accuracy']
    assert all(col in df.columns for col in expected_gps_columns)

def test_get_segment_analytics_df_timestamps(sample_campaign_zip):
    """Test timestamp handling in DataFrame conversion."""
    reader = CampaignReader(sample_campaign_zip, require_campaign_metadata=True)
    
    # Test with system time index
    df_system = reader.get_segment_analytics_df("test-segment", system_time_index=True)
    assert df_system.index.name == 'systemTime'
    assert isinstance(df_system.index, pd.DatetimeIndex)
    
    # Test with video time index
    df_video = reader.get_segment_analytics_df("test-segment", system_time_index=False)
    assert df_video.index.name == 'videoTime'
    assert df_video.index.dtype in [np.float64, np.int64]  # Should be numeric

def test_get_segment_location_summary(sample_campaign_zip):
    """Test location summary generation."""
    reader = CampaignReader(sample_campaign_zip, require_campaign_metadata=True)
    summary = reader.get_segment_location_summary("test-segment", min_accuracy=15.0)
    
    # Check summary contents
    assert summary['status'] == 'success'
    assert 'start_point' in summary
    assert 'end_point' in summary
    assert 'total_distance_meters' in summary
    assert 'average_speed_mps' in summary
    assert summary['valid_gps_points'] > 0
    
    # Check that distances make sense
    assert summary['total_distance_meters'] > 0
    assert summary['average_speed_mps'] >= 0

def test_get_segment_location_summary_bad_accuracy(sample_campaign_zip):
    """Test location summary with strict accuracy requirement."""
    reader = CampaignReader(sample_campaign_zip, require_campaign_metadata=True)
    summary = reader.get_segment_location_summary("test-segment", min_accuracy=1.0)
    
    # Should indicate insufficient data due to strict accuracy requirement
    assert summary['status'] == 'insufficient_data'
    assert summary['valid_gps_points'] == 0

def test_haversine_distances():
    """Test the Haversine distance calculation."""
    reader = CampaignReader("dummy.zip")  # Just for accessing the method
    
    # Test with known points
    lats = pd.Series([44.9553195, 44.9553295, 44.9553395])
    lons = pd.Series([-93.3773398, -93.3773298, -93.3773198])
    
    distances = reader._haversine_distances(lats, lons)
    
    assert len(distances) == len(lats)
    assert distances.iloc[0] == 0  # First point should have zero distance
    assert all(d >= 0 for d in distances)  # All distances should be non-negative

def test_get_combined_motion_location_analysis(sample_campaign_zip):
    """Test combined motion and location analysis."""
    reader = CampaignReader(sample_campaign_zip, require_campaign_metadata=True)
    combined = reader.get_combined_motion_location_analysis(
        "test-segment",
        motion_window="100ms",
        location_window="1S"
    )
    
    # Check that we have both motion and location data
    assert 'acceleration_magnitude' in combined.columns.get_level_values(0)
    assert 'angular_velocity_magnitude' in combined.columns.get_level_values(0)
    assert 'latitude' in combined.columns
    assert 'longitude' in combined.columns

def test_missing_gps_handling(sample_campaign_zip):
    """Test handling of missing GPS data."""
    reader = CampaignReader(sample_campaign_zip, require_campaign_metadata=True)
    df = reader.get_segment_analytics_df("test-segment")
    
    # Check that missing GPS data is represented as NaN
    assert df['latitude'].isna().any()
    assert df['longitude'].isna().any()
    assert df['accuracy'].isna().any()
    
    # Check that IMU data is still present when GPS is missing
    assert not df['acc_x'].isna().any()
    assert not df['acc_y'].isna().any()
    assert not df['acc_z'].isna().any()

def test_error_handling(sample_campaign_zip):
    """Test error handling for invalid data."""
    reader = CampaignReader(sample_campaign_zip, require_campaign_metadata=True)
    
    # Test with non-existent segment
    with pytest.raises(CampaignZipError):
        reader.get_segment_analytics_df("non-existent-segment")
    
    # Test with invalid segment ID
    with pytest.raises(CampaignZipError):
        reader.get_segment_location_summary("invalid-id")

def test_large_dataset_performance(sample_campaign_zip):
    """Test performance with a larger dataset."""
    reader = CampaignReader(sample_campaign_zip, require_campaign_metadata=True)
    
    # Time the DataFrame conversion
    import time
    start_time = time.time()
    df = reader.get_segment_analytics_df("test-segment")
    conversion_time = time.time() - start_time
    
    # Should process reasonably quickly
    assert conversion_time < 1.0  # Should take less than a second