import argparse
from pathlib import Path

import cv2
import pandas as pd

from campaign_reader import CampaignReader
from campaign_reader.analytics import AnalyticsAggregator


def load_campaign(campaign_path, sample_rate, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    with CampaignReader(campaign_path, "./campaign") as reader:
        campaign = reader.get_campaign_metadata()
        print(f"Campaign Name: {campaign.name}")
        print(f"Campaign Description: {campaign.description}")
        print(f"Campaign Created At: {campaign.created_at}")

        segment = reader.get_segments()[0]
        df = reader.get_segment_analytics_df(segment.id)

        # Basic statistics
        print("GPS Statistics:")
        print(df[['latitude', 'longitude']].describe())

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

        metadata = reader.get_segment_video_metadata(segment.id)

        print(f"Video Resolution: {metadata['video']['width']}x{metadata['video']['height']}")
        print(f"Duration: {metadata['duration']} seconds")
        print(f"Format: {metadata['format']}")

        df = segment.process_frames(
            sample_rate=sample_rate,
            output_dir=output_dir
        )
        print(f"Extracted {len(df)} frames")
        print(df.columns)
        print(df)
        # Access frame data
        for idx, row in df.iterrows():
            # Instead of accessing frame directly from DataFrame
            # we'll load it from the saved path
            frame_path = Path(row['frame_path'])

            # Clean up coordinate values for filename
            lat = f"{row['latitude']:.6f}" if pd.notna(row['latitude']) else "unknown"
            lon = f"{row['longitude']:.6f}" if pd.notna(row['longitude']) else "unknown"

            # Rename frame to include coordinates
            new_frame_path = output_dir / f"frame_{idx:04d}_{lat}_{lon}.jpg"

            if frame_path.exists():
                frame_path.rename(new_frame_path)
            else:
                print(f"Frame file not found: {frame_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from campaign video with analytics data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'campaign_path',
        help='Path to the campaign zip file'
    )
    
    parser.add_argument(
        '--sample-rate', '-s',
        type=float,
        default=30.0,
        help='Frame extraction sample rate (frames per second)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Output directory for extracted frames'
    )
    
    args = parser.parse_args()
    
    print(f"Processing campaign: {args.campaign_path}")
    print(f"Sample rate: {args.sample_rate} fps")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    load_campaign(args.campaign_path, args.sample_rate, args.output_dir)


if __name__ == '__main__':
    main()
