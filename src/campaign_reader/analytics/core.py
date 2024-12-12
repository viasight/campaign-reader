# analytics.py
from typing import List, Dict, Optional
import pandas as pd
import json
from pathlib import Path


class AnalyticsData:
    """Handles analytics data processing and validation."""

    def __init__(self, analytics_files: List[Path]):
        self.analytics_files = analytics_files
        self._dataframe: Optional[pd.DataFrame] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert analytics files to a pandas DataFrame with proper typing."""
        if self._dataframe is not None:
            return self._dataframe

        records = []
        for file in self.analytics_files:
            with open(file) as f:
                try:
                    # Load the entire file as JSON
                    data = json.load(f)
                    # If data is a list, process each record
                    if isinstance(data, list):
                        for record in data:
                            flat_record = self._flatten_record(record)
                            records.append(flat_record)
                    else:
                        # If it's a single record
                        flat_record = self._flatten_record(data)
                        records.append(flat_record)
                except json.JSONDecodeError:
                    # File might contain one JSON object per line
                    f.seek(0)  # Reset file pointer
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            flat_record = self._flatten_record(record)
                            records.append(flat_record)
                        except json.JSONDecodeError:
                            continue  # Skip invalid lines

        if not records:
            return pd.DataFrame()  # Return empty DataFrame if no valid records

        df = pd.DataFrame(records)

        # Convert timestamps
        if 'systemTime' in df.columns:
            # Convert to relative time since start of recording
            df['systemTime'] = df['systemTime'].astype(float)
            start_time = df['systemTime'].min()
            df['systemTime'] = (df['systemTime'] - start_time) / 1000.0  # Convert to seconds from start

        if 'videoTime' in df.columns:
            # Convert nanoseconds to seconds
            df['videoTime'] = df['videoTime'].astype(float) / 1e9

        # Ensure float type for numeric columns
        float_columns = ['latitude', 'longitude', 'accuracy',
                         'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z',
                         'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z']
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        self._dataframe = df
        return df

    @staticmethod
    def _flatten_record(record: Dict) -> Dict:
        """Flatten nested analytics record structure."""
        try:
            return {
                'index': record['index'],
                'systemTime': record['systemTime'],
                'videoTime': record['videoTime'],
                'latitude': record['gps']['latitude'],
                'longitude': record['gps']['longitude'],
                'accuracy': record['gps']['accuracy'],
                'linear_acceleration_x': record['imu']['linear_acceleration']['x'],
                'linear_acceleration_y': record['imu']['linear_acceleration']['y'],
                'linear_acceleration_z': record['imu']['linear_acceleration']['z'],
                'angular_velocity_x': record['imu']['angular_velocity']['x'],
                'angular_velocity_y': record['imu']['angular_velocity']['y'],
                'angular_velocity_z': record['imu']['angular_velocity']['z']
            }
        except (KeyError, TypeError) as e:
            # Return a record with None values if structure is invalid
            return {
                'index': None,
                'systemTime': None,
                'videoTime': None,
                'latitude': None,
                'longitude': None,
                'accuracy': None,
                'linear_acceleration_x': None,
                'linear_acceleration_y': None,
                'linear_acceleration_z': None,
                'angular_velocity_x': None,
                'angular_velocity_y': None,
                'angular_velocity_z': None
            }

        # analytics.py - updated validate() method only

    def validate(self) -> Dict[str, List[str]]:
        """Validate analytics data for consistency and completeness."""
        results = {
            'errors': [],
            'warnings': []
        }

        df = self.to_dataframe()
        if df.empty:
            results['errors'].append("No valid analytics data found")
            return results

        # Check for missing values
        missing = df.isnull().sum()
        missing_cols = [col for col, count in missing.items() if count > 0]
        if missing_cols:
            results['errors'].extend(
                f"Missing values in column '{col}': {missing[col]} records"
                for col in missing_cols
            )

        # Check timestamp gaps (now using numeric seconds)
        system_time_diff = df['systemTime'].diff()
        # Look for gaps >= 1 second
        large_gaps = system_time_diff[system_time_diff >= 1.0]
        if not large_gaps.empty:
            results['warnings'].append(f"Found {len(large_gaps)} time gaps >= 1 second in data collection")

        # Validate GPS coordinates
        invalid_latitudes = df[~df['latitude'].between(-90, 90)]['latitude']
        if not invalid_latitudes.empty:
            results['errors'].append("Invalid latitude values detected")

        invalid_longitudes = df[~df['longitude'].between(-180, 180)]['longitude']
        if not invalid_longitudes.empty:
            results['errors'].append("Invalid longitude values detected")

        return results