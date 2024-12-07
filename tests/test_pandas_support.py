def test_get_segment_analytics_df(test_campaign_zip):
    """Test getting analytics data as a DataFrame."""
    reader = CampaignReader(str(test_campaign_zip))
    df = reader.get_segment_analytics_df("test-segment")
    
    # Basic DataFrame checks
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    
    # Check column presence and types
    assert 'system_time' in df.columns
    assert 'video_time' in df.columns
    assert 'gps_latitude' in df.columns
    assert 'imu_linear_acceleration_x' in df.columns
    
    assert isinstance(df['system_time'].iloc[0], pd.Timestamp)
    assert isinstance(df['video_time'].iloc[0], (int, np.int64, float, np.float64))
    
    # Check data values
    assert df['gps_latitude'].iloc[0] == 44.9553195
    assert df['gps_accuracy'].iloc[0] == 14.813
    assert abs(df['imu_linear_acceleration_z'].iloc[0] - 9.778867721557617) < 1e-10

def test_get_segment_analytics_df_unflattened(test_campaign_zip):
    """Test getting analytics data without flattening nested structures."""
    reader = CampaignReader(str(test_campaign_zip))
    df = reader.get_segment_analytics_df("test-segment", flatten=False)
    
    # Check that GPS and IMU are still dictionary columns
    assert 'gps' in df.columns
    assert 'imu' in df.columns
    assert isinstance(df['gps'].iloc[0], dict)
    assert isinstance(df['imu'].iloc[0], dict)

def test_get_segment_analytics_df_time_filtering(test_campaign_zip):
    """Test time-based filtering of analytics data."""
    reader = CampaignReader(str(test_campaign_zip))
    
    # Test start_time filtering
    start_time = pd.to_datetime(1733327149360, unit='ms')
    df = reader.get_segment_analytics_df("test-segment", start_time=start_time)
    assert len(df) == 1
    assert df['index'].iloc[0] == 1
    
    # Test end_time filtering
    end_time = pd.to_datetime(1733327149360, unit='ms')
    df = reader.get_segment_analytics_df("test-segment", end_time=end_time)
    assert len(df) == 1
    assert df['index'].iloc[0] == 0

def test_get_analytics_summary(test_campaign_zip):
    """Test analytics summary generation."""
    reader = CampaignReader(str(test_campaign_zip))
    summary_df = reader.get_analytics_summary()
    
    # Basic checks
    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 1
    
    # Check summary values
    assert summary_df['segment_id'].iloc[0] == 'test-segment'
    assert summary_df['data_points'].iloc[0] == 2
    assert summary_df['gps_accuracy_mean'].iloc[0] == 14.813
    assert summary_df['gps_points'].iloc[0] == 2

def test_error_handling(test_campaign_zip):
    """Test error handling for invalid segments and data."""
    reader = CampaignReader(str(test_campaign_zip))
    
    # Invalid segment ID
    df = reader.get_segment_analytics_df("nonexistent-segment")
    assert df.empty
    
    # Invalid campaign zip
    with pytest.raises(Exception):
        CampaignReader("nonexistent.zip")