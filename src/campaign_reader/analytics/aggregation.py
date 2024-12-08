from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


class AnalyticsAggregator:
    """
    Provides aggregation and summary statistics for campaign analytics data.
    """

    def __init__(self, analytics_df: pd.DataFrame):
        """
        Initialize the aggregator with a DataFrame containing analytics data.

        Args:
            analytics_df: DataFrame containing analytics data from campaign segments
        """
        self.df = analytics_df
        self._validate_dataframe()

    def _validate_dataframe(self) -> None:
        """Validates that the DataFrame has the required columns."""
        required_columns = ['index', 'systemTime', 'videoTime',
                            'latitude', 'longitude', 'accuracy']
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _ensure_datetime(self) -> None:
        """Ensures datetime column exists in the DataFrame."""
        if 'datetime' not in self.df.columns:
            if not isinstance(self.df['systemTime'].iloc[0], pd.Timestamp):
                self.df['datetime'] = pd.to_datetime(self.df['systemTime'].astype(float), unit='ms')
            else:
                self.df['datetime'] = self.df['systemTime']

    def get_temporal_stats(self) -> Dict[str, float]:
        """
        Calculate temporal statistics for the campaign.

        Returns:
            Dictionary containing temporal statistics:
            - total_duration: Total duration in seconds
            - avg_sampling_rate: Average samples per second
            - start_time: Start timestamp
            - end_time: End timestamp
        """
        self._ensure_datetime()

        # Calculate duration in seconds
        duration = (self.df['datetime'].max() - self.df['datetime'].min()).total_seconds()

        # For sample rate, we use (n-1) for duration since we want the rate between samples
        # n samples span (n-1) intervals
        stats = {
            'total_duration': duration,
            'avg_sampling_rate': (len(self.df) - 1) / duration if duration > 0 else 0,
            'start_time': self.df['datetime'].min().isoformat(),
            'end_time': self.df['datetime'].max().isoformat()
        }
        return stats

    def get_spatial_stats(self) -> Dict[str, float]:
        """
        Calculate spatial statistics for the GPS data.

        Returns:
            Dictionary containing spatial statistics:
            - total_distance: Total distance traveled in meters
            - avg_speed: Average speed in meters per second
            - max_speed: Maximum speed in meters per second
            - min_accuracy: Best GPS accuracy in meters
            - max_accuracy: Worst GPS accuracy in meters
        """
        from geopy.distance import geodesic

        self._ensure_datetime()

        # Calculate distances between consecutive points
        coords = list(zip(self.df['latitude'], self.df['longitude']))
        distances = [
            geodesic(coords[i], coords[i + 1]).meters
            for i in range(len(coords) - 1)
        ]

        # Calculate speeds using time differences
        time_diffs = np.diff(self.df['datetime'].astype(np.int64)) / 1e9  # Convert to seconds
        speeds = [d / t for d, t in zip(distances, time_diffs) if t > 0]

        return {
            'total_distance': sum(distances),
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': max(speeds) if speeds else 0,
            'min_accuracy': self.df['accuracy'].min(),
            'max_accuracy': self.df['accuracy'].max()
        }

    def get_motion_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for IMU data.

        Returns:
            Dictionary containing IMU statistics for linear acceleration
            and angular velocity across all axes.
        """
        imu_stats = {
            'linear_acceleration': {},
            'angular_velocity': {}
        }

        # Linear acceleration stats
        for axis in ['x', 'y', 'z']:
            col = f'linear_acceleration_{axis}'
            if col in self.df.columns:
                imu_stats['linear_acceleration'][axis] = {
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }

        # Angular velocity stats
        for axis in ['x', 'y', 'z']:
            col = f'angular_velocity_{axis}'
            if col in self.df.columns:
                imu_stats['angular_velocity'][axis] = {
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max()
                }

        return imu_stats

    def get_summary(self) -> Dict[str, Union[Dict, float, str]]:
        """
        Get a comprehensive summary of all analytics data.

        Returns:
            Dictionary containing all summary statistics.
        """
        return {
            'temporal': self.get_temporal_stats(),
            'spatial': self.get_spatial_stats(),
            'motion': self.get_motion_stats()
        }

    def aggregate_by_interval(self,
                              interval: str = '1S',
                              metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregate data by time interval.

        Args:
            interval: Pandas offset string (e.g., '1S' for 1 second, '5T' for 5 minutes)
            metrics: List of columns to aggregate. If None, aggregates all numeric columns.

        Returns:
            DataFrame with aggregated statistics
        """
        if metrics is None:
            metrics = self.df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove index from aggregation metrics if present
            if 'index' in metrics:
                metrics.remove('index')

        self._ensure_datetime()

        # Create a copy with datetime index
        df_temp = self.df.copy()
        df_temp.set_index('datetime', inplace=True)

        # Define aggregation functions for different column types
        agg_funcs = {
            col: ['mean', 'std', 'min', 'max']
            for col in metrics
        }

        # Perform aggregation
        return df_temp[metrics].resample(interval).agg(agg_funcs)