            if not self.require_campaign_metadata:
                self._load_campaign()
        return self._campaign
    
    def get_segment(self, segment_id: str) -> Optional[CampaignSegment]:
        """Get a specific segment by ID."""
        campaign = self.get_campaign_metadata()
        return campaign.get_segment(segment_id) if campaign else None
    
    def get_segments(self) -> List[CampaignSegment]:
        """Get all segments in sequence order."""
        campaign = self.get_campaign_metadata()
        return campaign.get_ordered_segments() if campaign else []
    
    def iter_segments(self) -> Generator[CampaignSegment, None, None]:
        """Iterate through segments in sequence order."""
        campaign = self.get_campaign_metadata()
        if campaign:
            for segment in campaign.get_ordered_segments():
                yield segment
    
    def get_segment_analytics(self, segment_id: str) -> List[dict]:
        """Get analytics data for a specific segment."""
        segment = self.get_segment(segment_id)
        if not segment:
            raise CampaignZipError(f"Segment {segment_id} not found")
            
        analytics_dir = segment.get_analytics_path()
        if not analytics_dir or not analytics_dir.exists():
            raise CampaignZipError(f"Analytics directory not found for segment {segment_id}")
            
        analytics_data = []
        try:
            for analytics_file in sorted(analytics_dir.glob('analytics*.json')):
                with open(analytics_file) as f:
                    analytics_data.extend(json.load(f))
        except Exception as e:
            raise CampaignZipError(f"Failed to load analytics data: {str(e)}")
            
        return analytics_data
    
    def get_segment_analytics_df(self, segment_id: str, 
                               start_time: Optional[Union[datetime, int]] = None,
                               end_time: Optional[Union[datetime, int]] = None,
                               flatten: bool = True) -> pd.DataFrame:
        """Get analytics data for a segment as a pandas DataFrame.
        
        Args:
            segment_id (str): ID of the segment to get analytics for
            start_time (Optional[Union[datetime, int]]): Filter data after this time
                Can be datetime object or Unix timestamp in milliseconds
            end_time (Optional[Union[datetime, int]]): Filter data before this time
                Can be datetime object or Unix timestamp in milliseconds
            flatten (bool): If True, flattens nested JSON structure into columns
                If False, keeps GPS and IMU data in dictionary columns
                
        Returns:
            pd.DataFrame: DataFrame containing analytics data with columns described in docstring
        """
        try:
            analytics_data = self.get_segment_analytics(segment_id)
            if not analytics_data:
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(analytics_data)
            
            # Rename columns to be more Python-friendly
            df = df.rename(columns={
                'systemTime': 'system_time',
                'videoTime': 'video_time'
            })
            
            # Convert timestamps
            df['system_time'] = pd.to_datetime(df['system_time'].astype(np.int64), unit='ms')
            df['video_time'] = pd.to_numeric(df['video_time'])
            
            if flatten:
                # Extract nested GPS data
                df['gps_latitude'] = df['gps'].apply(lambda x: x.get('latitude'))
                df['gps_longitude'] = df['gps'].apply(lambda x: x.get('longitude'))
                df['gps_accuracy'] = df['gps'].apply(lambda x: x.get('accuracy'))
                
                # Extract nested IMU data
                df['imu_linear_acceleration_x'] = df['imu'].apply(lambda x: x.get('linear_acceleration', {}).get('x'))
                df['imu_linear_acceleration_y'] = df['imu'].apply(lambda x: x.get('linear_acceleration', {}).get('y'))
                df['imu_linear_acceleration_z'] = df['imu'].apply(lambda x: x.get('linear_acceleration', {}).get('z'))
                
                df['imu_angular_velocity_x'] = df['imu'].apply(lambda x: x.get('angular_velocity', {}).get('x'))
                df['imu_angular_velocity_y'] = df['imu'].apply(lambda x: x.get('angular_velocity', {}).get('y'))
                df['imu_angular_velocity_z'] = df['imu'].apply(lambda x: x.get('angular_velocity', {}).get('z'))
                
                # Drop original nested columns
                df = df.drop(columns=['gps', 'imu'])
            
            # Apply time filters if provided
            if start_time is not None:
                if isinstance(start_time, int):
                    start_time = pd.to_datetime(start_time, unit='ms')
                df = df[df['system_time'] >= start_time]
                
            if end_time is not None:
                if isinstance(end_time, int):
                    end_time = pd.to_datetime(end_time, unit='ms')
                df = df[df['system_time'] <= end_time]
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load analytics data for segment {segment_id}: {str(e)}")
            return pd.DataFrame()