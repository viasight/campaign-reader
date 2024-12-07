import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import zipfile
import tempfile
import os

from campaign_reader import CampaignReader

@pytest.fixture
def sample_analytics_data():
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