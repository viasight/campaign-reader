    def get_campaign_analytics_df(self, 
                                start_time: Optional[Union[datetime, int]] = None,
                                end_time: Optional[Union[datetime, int]] = None,
                                flatten: bool = True) -> pd.DataFrame:
        """Get analytics data for all segments in the campaign as a single DataFrame.
        
        The resulting DataFrame includes a 'segment_id' column to identify the source segment.
        
        Args:
            start_time (Optional[Union[datetime, int]]): Filter data after this time
                Can be datetime object or Unix timestamp in milliseconds
            end_time (Optional[Union[datetime, int]]): Filter data before this time
                Can be datetime object or Unix timestamp in milliseconds
            flatten (bool): If True, flattens nested JSON structure
                
        Returns:
            pd.DataFrame: DataFrame containing all segments' analytics data
        """
        dfs = []
        for segment in self.get_segments():
            try:
                segment_df = self.get_segment_analytics_df(
                    segment.id, 
                    start_time=start_time, 
                    end_time=end_time,
                    flatten=flatten
                )
                if not segment_df.empty:
                    # Add segment metadata
                    segment_df['segment_id'] = segment.id
                    segment_df['sequence_number'] = segment.sequence_number
                    segment_df['recorded_at'] = segment.recorded_at
                    dfs.append(segment_df)
            except Exception as e:
                logger.warning(f"Failed to load analytics for segment {segment.id}: {str(e)}")
                continue
                
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)
    
    def get_analytics_summary(self) -> pd.DataFrame:
        """Get a summary of analytics data across all segments.
        
        Returns a DataFrame with one row per segment containing:
        - Segment ID and sequence number
        - Start and end times
        - Number of data points
        - Basic statistics for numeric columns
        
        Returns:
            pd.DataFrame: Summary DataFrame
        """
        summaries = []
        
        for segment in self.get_segments():
            try:
                df = self.get_segment_analytics_df(segment.id)
                if df.empty:
                    continue
                    
                summary = {
                    'segment_id': segment.id,
                    'sequence_number': segment.sequence_number,
                    'start_time': df['system_time'].min(),
                    'end_time': df['system_time'].max(),
                    'duration_seconds': (df['system_time'].max() - df['system_time'].min()).total_seconds(),
                    'data_points': len(df),
                }
                
                # Add basic stats for numeric columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                for col in numeric_cols:
                    if col != 'sequence_number':
                        summary[f'{col}_mean'] = df[col].mean()
                        summary[f'{col}_std'] = df[col].std()
                        summary[f'{col}_min'] = df[col].min()
                        summary[f'{col}_max'] = df[col].max()
                        
                summaries.append(summary)
                
            except Exception as e:
                logger.warning(f"Failed to summarize analytics for segment {segment.id}: {str(e)}")
                continue
                
        if not summaries:
            return pd.DataFrame()
            
        return pd.DataFrame(summaries)
    
    def get_extracted_file(self, filename: str) -> Optional[Path]:
        """Get path to an extracted file."""
        return self._extracted_files.get(filename)
    
    def list_files(self) -> List[str]:
        """Get a list of all files in the zip."""
        return self.file_list
    
    def cleanup(self) -> None:
        """Remove all extracted files and directories."""
        if self._extract_dir.exists():
            try:
                # If directory is readonly, try to make it writable first
                current_mode = self._extract_dir.stat().st_mode
                if not os.access(self._extract_dir, os.W_OK):
                    os.chmod(self._extract_dir, current_mode | 0o700)

                # Remove files
                for file_path in self._extracted_files.values():
                    if file_path.exists():
                        file_path.unlink(missing_ok=True)
                
                # Remove empty directories, but only if we created the temp dir
                if self._using_temp_dir:
                    for dir_path in sorted(self._extract_dir.rglob('*'), reverse=True):
                        if dir_path.is_dir():
                            try:
                                dir_path.rmdir()
                            except OSError:
                                pass
                    
                    try:
                        self._extract_dir.rmdir()
                    except OSError:
                        pass

                # Restore original permissions if we changed them
                if not os.access(self._extract_dir, os.W_OK):
                    os.chmod(self._extract_dir, current_mode)
                    
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()