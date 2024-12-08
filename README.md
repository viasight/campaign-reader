# Campaign Reader

A Python package for reading and analyzing video campaign data stored in ZIP archives. The package provides tools for handling campaign metadata, video segments, and analytics data with features for validation and analysis.

## Features

- **Campaign Structure Handling**
  - ZIP file validation and extraction
  - Campaign metadata parsing
  - Segment access and ordering
  - Resource cleanup and management

- **Analytics Processing**
  - Conversion to pandas DataFrame
  - Time series analytics data validation
  - GPS coordinate validation
  - Gap detection in time series data
  - Statistical aggregation and analysis
  - Time-based data resampling

- **Video Metadata**
  - Video file information extraction
  - Codec and format details
  - Resolution and framerate analysis
  - Size and duration information

## Installation

```bash
pip install campaign-reader
```

### Dependencies

- Python 3.8+
- pandas
- ffmpeg (optional, for video metadata extraction)
- geopy (for GPS distance calculations)

## Usage

### Basic Campaign Reading

```python
from campaign_reader import CampaignReader

# Open a campaign zip file
with CampaignReader("campaign.zip") as reader:
    # Get campaign metadata
    campaign = reader.get_campaign_metadata()
    print(f"Campaign: {campaign.name}")
    
    # Iterate through segments
    for segment in reader.iter_segments():
        print(f"Segment {segment.sequence_number}: {segment.id}")
```

### Working with Analytics Data

```python
with CampaignReader("campaign.zip") as reader:
    # Get segment analytics as DataFrame
    segment = reader.get_segments()[0]
    df = reader.get_segment_analytics_df(segment.id)
    
    # Basic statistics
    print("GPS Statistics:")
    print(df[['latitude', 'longitude']].describe())
    
    # Validate analytics data
    validation = reader.validate_segment_analytics(segment.id)
    if validation['errors']:
        print("Validation errors:", validation['errors'])
    if validation['warnings']:
        print("Validation warnings:", validation['warnings'])
```

### Analytics Aggregation and Analysis

```python
from campaign_reader.analytics import AnalyticsAggregator

with CampaignReader("campaign.zip") as reader:
    # Get analytics data
    df = reader.get_segment_analytics_df(segment_id)
    
    # Create aggregator
    aggregator = AnalyticsAggregator(df)
    
    # Get comprehensive statistics
    summary = aggregator.get_summary()
    print(f"Total Distance: {summary['spatial']['total_distance']:.2f} meters")
    print(f"Average Speed: {summary['spatial']['avg_speed']:.2f} m/s")
    print(f"Duration: {summary['temporal']['total_duration']:.2f} seconds")
    
    # Aggregate data by time interval
    resampled = aggregator.aggregate_by_interval('5S')  # 5-second intervals
    print("\nIMU Statistics by 5-second intervals:")
    print(resampled['linear_acceleration_z'].describe())
```

### Extracting Video Metadata

```python
with CampaignReader("campaign.zip") as reader:
    segment = reader.get_segments()[0]
    metadata = reader.get_segment_video_metadata(segment.id)
    
    print(f"Video Resolution: {metadata['video']['width']}x{metadata['video']['height']}")
    print(f"Duration: {metadata['duration']} seconds")
    print(f"Format: {metadata['format']}")
```

## Campaign File Structure

The package expects campaign ZIP files with the following structure:

```
campaign.zip
├── metadata
│   └── campaign.json
└── segments
    ├── {segment-uuid}
    │   ├── analytics
    │   │   ├── analytics.json
    │   │   ├── analytics_1.json
    │   │   └── analytics_{n}.json
    │   └── video.mp4
    └── {segment-uuid}
        ├── analytics/
        └── video.mp4
```

### Campaign Metadata Format

```json
{
  "id": "060953aa-7baf-435b-aee2-2faaeb438aaf",
  "name": "Test Campaign",
  "createdAt": 1733434689359,
  "description": "Campaign description",
  "segments": [
    {
      "id": "7e0226fd-2903-4d4b-b399-f103e9063e06",
      "sequenceNumber": 0,
      "recordedAt": 1733434791634,
      "videoPath": "/data/video.mp4",
      "analyticsFilePattern": "/data/analytics_{}"
    }
  ]
}
```

### Analytics Data Format

```json
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
}
```

## Analytics Aggregation

The package provides comprehensive analytics aggregation through the `AnalyticsAggregator` class:

- **Temporal Statistics**
  - Total duration
  - Average sampling rate
  - Start and end timestamps

- **Spatial Statistics**
  - Total distance traveled
  - Average and maximum speeds
  - GPS accuracy ranges

- **Motion Statistics**
  - IMU data analysis (acceleration and angular velocity)
  - Statistical summaries (mean, std, min, max)
  - Time-based aggregation and resampling

## Error Handling

The package provides specific error types for different failure scenarios:

- `CampaignZipError`: Base exception for campaign zip file errors
- `FileNotFoundError`: When the campaign file doesn't exist
- `ValueError`: For video metadata extraction failures

Example error handling:

```python
try:
    with CampaignReader("campaign.zip") as reader:
        metadata = reader.get_segment_video_metadata(segment_id)
except CampaignZipError as e:
    print(f"Campaign error: {e}")
except ValueError as e:
    print(f"Video metadata error: {e}")
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[MIT License](LICENSE)
