from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from pathlib import Path

@dataclass
class CampaignSegment:
    """Represents a segment in a campaign."""
    id: str
    sequence_number: int
    recorded_at: datetime
    video_path: str
    analytics_file_pattern: str
    _extracted_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'CampaignSegment':
        """Create a segment from a dictionary."""
        return cls(
            id=data['id'],
            sequence_number=data['sequenceNumber'],
            recorded_at=datetime.fromtimestamp(data['recordedAt'] / 1000.0),
            video_path=data['videoPath'],
            analytics_file_pattern=data['analyticsFilePattern']
        )

    def get_video_path(self) -> Optional[Path]:
        """Get the path to the extracted video file."""
        if self._extracted_path:
            return self._extracted_path / 'video.mp4'
        return None

    def get_analytics_path(self) -> Optional[Path]:
        """Get the path to the analytics directory."""
        if self._extracted_path:
            return self._extracted_path / 'analytics'
        return None

@dataclass
class Campaign:
    """Represents a campaign with metadata and segments."""
    id: str
    name: str
    created_at: datetime
    description: str
    segments: List[CampaignSegment]

    @classmethod
    def from_dict(cls, data: dict) -> 'Campaign':
        """Create a campaign from a dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            created_at=datetime.fromtimestamp(data['createdAt'] / 1000.0),
            description=data['description'],
            segments=[CampaignSegment.from_dict(seg) for seg in data['segments']]
        )

    def get_segment(self, segment_id: str) -> Optional[CampaignSegment]:
        """Get a segment by its ID."""
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        return None

    def get_ordered_segments(self) -> List[CampaignSegment]:
        """Get segments ordered by sequence number."""
        return sorted(self.segments, key=lambda x: x.sequence_number)