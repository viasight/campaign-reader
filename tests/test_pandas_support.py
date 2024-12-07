@pytest.fixture
def sample_campaign_json():
    return {
        "id": "test-campaign",
        "name": "Test Campaign",
        "createdAt": 1733327149354,
        "description": "Test campaign for unit tests",
        "segments": [
            {
                "id": "test-segment",
                "sequenceNumber": 0,
                "recordedAt": 1733327149354,
                "videoPath": "/data/video.mp4",
                "analyticsFilePattern": "/data/analytics_{}.json"
            }
        ]
    }

@pytest.fixture
def test_campaign_zip(tmp_path, sample_analytics_data, sample_campaign_json):
    """Create a test campaign zip file with sample data."""
    # Create directory structure
    campaign_dir = tmp_path / "campaign"
    metadata_dir = campaign_dir / "metadata"
    segment_dir = campaign_dir / "segments" / "test-segment" / "analytics"
    metadata_dir.mkdir(parents=True)
    segment_dir.mkdir(parents=True)
    
    # Write campaign metadata
    with open(metadata_dir / "campaign.json", "w") as f:
        json.dump(sample_campaign_json, f)
    
    # Write analytics data
    with open(segment_dir / "analytics.json", "w") as f:
        json.dump(sample_analytics_data, f)
    
    # Create zip file
    zip_path = tmp_path / "test_campaign.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, _, files in os.walk(campaign_dir):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(campaign_dir)
                zf.write(file_path, arc_path)
    
    return zip_path