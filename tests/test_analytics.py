import json

import pandas as pd
import pytest

from campaign_reader.analytics import AnalyticsData


@pytest.fixture
def sample_analytics_json():
    """Sample analytics data in the new format."""
    return [
        {
            "index": 0,
            "systemTime": "1733327149354",
            "videoTime": "0",
            "gps": {
                "latitude": 44.9553195,
                "longitude": -93.3773398,
                "accuracy": 14.813
            },
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
        },
        {
            "index": 1,
            "systemTime": "1733327149363",
            "videoTime": "33478000",
            "gps": {
                "latitude": 44.9553195,
                "longitude": -93.3773398,
                "accuracy": 14.813
            },
            "imu": {
                "linear_acceleration": {
                    "x": -0.017946170642971992,
                    "y": 0.2709871530532837,
                    "z": 9.771689414978027
                },
                "angular_velocity": {
                    "x": -3.054326225537807E-4,
                    "y": 0.0012217304902151227,
                    "z": 1.5271631127689034E-4
                }
            }
        }
    ]


@pytest.fixture
def analytics_files(tmp_path, sample_analytics_json):
    """Create temporary analytics files."""
    files = []
    for i in range(3):
        file_path = tmp_path / f'analytics_{i}.json'
        with open(file_path, 'w') as f:
            json.dump(sample_analytics_json, f)
        files.append(file_path)
    return files


@pytest.fixture
def analytics_data(analytics_files):
    """Create AnalyticsData instance with sample files."""
    return AnalyticsData(analytics_files)


def test_analytics_to_dataframe(analytics_data):
    """Test conversion of analytics data to DataFrame."""
    df = analytics_data.to_dataframe()

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6  # 3 files × 2 records each

    # Check column types
    assert pd.api.types.is_datetime64_any_dtype(df['systemTime'])
    assert pd.api.types.is_timedelta64_dtype(df['videoTime'])
    assert pd.api.types.is_float_dtype(df['latitude'])
    assert pd.api.types.is_float_dtype(df['linear_acceleration_x'])

    # Check values
    assert df['latitude'].iloc[0] == 44.9553195
    assert df['longitude'].iloc[0] == -93.3773398
    assert df['linear_acceleration_x'].iloc[0] == pytest.approx(-0.020339)


def test_analytics_validation_valid_data(analytics_data):
    """Test validation with valid analytics data."""
    validation_results = analytics_data.validate()

    assert 'errors' in validation_results
    assert 'warnings' in validation_results
    assert len(validation_results['errors']) == 0

    # Might have a warning about timestamp gaps, which is expected
    if validation_results['warnings']:
        assert any('gaps' in warning for warning in validation_results['warnings'])


def test_analytics_validation_invalid_data(tmp_path):
    """Test validation with invalid analytics data."""
    # Create file with invalid data
    invalid_data = [
        {
            "index": 0,
            "systemTime": "1733327149354",
            "videoTime": "0",
            "gps": {
                "latitude": 91.0,  # Invalid latitude
                "longitude": -93.3773398,
                "accuracy": 14.813
            },
            "imu": {
                "linear_acceleration": {"x": 0, "y": 0, "z": 0},
                "angular_velocity": {"x": 0, "y": 0, "z": 0}
            }
        }
    ]

    file_path = tmp_path / 'invalid_analytics.json'
    with open(file_path, 'w') as f:
        json.dump(invalid_data, f)

    analytics = AnalyticsData([file_path])
    validation_results = analytics.validate()

    assert any('Invalid latitude' in error for error in validation_results['errors'])


def test_analytics_with_missing_values(tmp_path):
    """Test handling of analytics data with missing values."""
    incomplete_data = [
        {
            "index": 0,
            "systemTime": "1733327149354",
            "videoTime": "0",
            "gps": {
                "latitude": None,  # Missing value
                "longitude": -93.3773398,
                "accuracy": 14.813
            },
            "imu": {
                "linear_acceleration": {"x": 0, "y": 0, "z": 0},
                "angular_velocity": {"x": 0, "y": 0, "z": 0}
            }
        }
    ]

    file_path = tmp_path / 'incomplete_analytics.json'
    with open(file_path, 'w') as f:
        json.dump(incomplete_data, f)

    analytics = AnalyticsData([file_path])
    validation_results = analytics.validate()

    assert any('Missing values' in error for error in validation_results['errors'])