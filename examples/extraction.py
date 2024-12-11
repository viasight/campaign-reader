from pathlib import Path

import cv2
import pandas as pd

from campaign_reader import CampaignReader
from campaign_reader.analytics import AnalyticsAggregator


def load_campaign(name):

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    with CampaignReader(name, "./campaign") as reader:
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

        frame_data = segment.extract_frames(sample_rate=5.0)  # 5 frames per second

        # Convert to DataFrame with aligned data
        df = frame_data.to_dataframe()
        print(f"Extracted {len(df)} frames")
        print(df.columns)
        print(df)
        # Access frame data
        for idx, row in df.iterrows():
            frame = row['frame']  # numpy array containing the image

            # Clean up coordinate values for filename
            lat = f"{row['latitude_x']:.6f}" if pd.notna(row['latitude_x']) else "unknown"
            lon = f"{row['longitude_x']:.6f}" if pd.notna(row['longitude_x']) else "unknown"

            # Construct frame path
            frame_path = output_dir / f"frame_{idx:04d}_{lat}_{lon}.jpg"
            print(f"Saving frame {frame_path}")

            try:
                # Save the frame directly without color space conversion first
                success = cv2.imwrite(str(frame_path), frame)
                if not success:
                    print(f"Failed to write frame {frame_path}")
                    # Try with color conversion if direct save fails
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(str(frame_path), frame_bgr)
                    if not success:
                        print(f"Failed to write frame {frame_path} even after color conversion")
            except Exception as e:
                print(f"Error saving frame {frame_path}: {str(e)}")

            # Print frame info for debugging
            if idx == 0:  # Print info for first frame
                print(f"Frame shape: {frame.shape}")
                print(f"Frame dtype: {frame.dtype}")
                print(f"Frame min/max values: {frame.min()}, {frame.max()}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    load_campaign('/Users/zikomofields/Downloads/Lake Lotawana HOA Roads_1733949339302.zip')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
