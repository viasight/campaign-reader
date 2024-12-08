# tests/test_analytics_aggregation.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from campaign_reader.analytics.aggregation import AnalyticsAggregator


@pytest.fixture
def sample_analytics_df():
    """Create a sample DataFrame for testing."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    data = []

    for i in range(10):
        timestamp = base_time + timedelta(seconds=i)
        data.append({
            'index': i,
            'systemTime': str(int(timestamp.timestamp() * 1000)),
            'videoTime': str(i),
            'latitude': 44.9553195 + (i * 0.0001),
            'longitude': -93.3773398 + (i * 0.0001),
            'accuracy': 14.813 + (i * 0.1),
            'linear_acceleration_x': -0.02 + (i * 0.01),
            'linear_acceleration_y': 0.26 + (i * 0.01),
            'linear_acceleration_z': 9.77 + (i * 0.01),
            'angular_velocity_x': -0.001 + (i * 0.001),
            'angular_velocity_y': -0.002 + (i * 0.001),
            'angular_velocity_z': 0.0001 + (i * 0.001)
        })

    return pd.DataFrame(data)


def test_init_validation(sample_analytics_df):
    """Test that initialization validates required columns."""
    # Should work with valid DataFrame
    aggregator = AnalyticsAggregator(sample_analytics_df)
    assert isinstance(aggregator, AnalyticsAggregator)

    # Should raise error with missing columns
    invalid_df = sample_analytics_df.drop(columns=['latitude'])
    with pytest.raises(ValueError):
        AnalyticsAggregator(invalid_df)


def test_temporal_stats(sample_analytics_df):
    """Test temporal statistics calculation."""
    aggregator = AnalyticsAggregator(sample_analytics_df)
    stats = aggregator.get_temporal_stats()

    assert isinstance(stats, dict)
    assert 'total_duration' in stats
    assert 'avg_sampling_rate' in stats
    assert stats['total_duration'] == pytest.approx(9.0)  # 10 samples, 1 second apart
    assert stats['avg_sampling_rate'] == pytest.approx(1.0)  # 1 sample per second

def test_spatial_stats(sample_analytics_df):
    """Test spatial statistics calculation."""
    aggregator = AnalyticsAggregator(sample_analytics_df)
    stats = aggregator.get_spatial_stats()

    assert isinstance(stats, dict)
    assert 'total_distance' in stats
    assert 'avg_speed' in stats
    assert 'min_accuracy' in stats
    assert stats['min_accuracy'] == pytest.approx(14.813)
    assert stats['max_accuracy'] == pytest.approx(14.813 + 0.9)


def test_motion_stats(sample_analytics_df):
    """Test IMU statistics calculation."""
    aggregator = AnalyticsAggregator(sample_analytics_df)
    stats = aggregator.get_motion_stats()

    assert isinstance(stats, dict)
    assert 'linear_acceleration' in stats
    assert 'angular_velocity' in stats
    assert 'x' in stats['linear_acceleration']
    assert 'mean' in stats['linear_acceleration']['x']


def test_aggregate_by_interval(sample_analytics_df):
    """Test time-based aggregation."""
    aggregator = AnalyticsAggregator(sample_analytics_df)
    aggregated = aggregator.aggregate_by_interval('5S')

    assert isinstance(aggregated, pd.DataFrame)
    assert len(aggregated) == 2  # Should have 2 5-second intervals
    assert isinstance(aggregated.index, pd.DatetimeIndex)


def test_get_summary(sample_analytics_df):
    """Test comprehensive summary generation."""
    aggregator = AnalyticsAggregator(sample_analytics_df)
    summary = aggregator.get_summary()

    assert isinstance(summary, dict)
    assert 'temporal' in summary
    assert 'spatial' in summary
    assert 'motion' in summary