# tests/conftest.py

import json
import os
import subprocess
import sys
import zipfile

import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


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
            "systemTime": "1733327150354",  # 1 second gap
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
def campaign_zip(sample_campaign_data, sample_analytics_data, test_video_data, tmp_path):
    """Create a temporary campaign zip file with test content."""
    zip_path = tmp_path / 'test_campaign.zip'

    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add campaign metadata
        zf.writestr('metadata/campaign.json', json.dumps(sample_campaign_data))

        # Add segment files
        for segment in sample_campaign_data['segments']:
            # Add video file with proper test data
            segment_path = f"segments/{segment['id']}/video.mp4"
            zf.writestr(segment_path, test_video_data)

            # Add analytics files
            analytics_path = f"segments/{segment['id']}/analytics/analytics.json"
            zf.writestr(analytics_path, json.dumps(sample_analytics_data))

            # Add additional analytics files
            for i in range(1, 3):
                analytics_path = f"segments/{segment['id']}/analytics/analytics_{i}.json"
                zf.writestr(analytics_path, json.dumps(sample_analytics_data))

    yield str(zip_path)


@pytest.fixture
def test_video_data(tmp_path):
    """Create a test video file."""
    video_path = tmp_path / 'test.mp4'
    try:
        # Create a 1-second black video
        subprocess.run([
            'ffmpeg',
            '-f', 'lavfi',
            '-i', 'color=c=black:s=320x240:r=30:d=1',
            '-c:v', 'libx264',
            '-t', '1',
            str(video_path)
        ], check=True, capture_output=True)
        with open(video_path, 'rb') as f:
            return f.read()
    except subprocess.CalledProcessError:
        # If ffmpeg fails, return a minimal MP4 file structure
        return (
            b'\x00\x00\x00\x20ftypisom'  # File type box
            b'\x00\x00\x00\x08free'  # Free space box
            b'\x00\x00\x00\x08mdat'  # Media data box
        )
