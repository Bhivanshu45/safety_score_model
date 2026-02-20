"""
DateTime Utilities
Extract temporal features from datetime for model prediction
"""

from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def extract_temporal_features(dt: datetime) -> Dict[str, int]:
    """
    Extract temporal features from datetime object
    
    Args:
        dt: datetime object
        
    Returns:
        Dictionary with temporal features:
        - time_bin: 4-hour bin (0-5)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - month: Month (1-12)
        - year: Year
        - is_weekend: Weekend indicator (0=Weekday, 1=Weekend)
    """
    
    # Extract basic components
    hour = dt.hour
    day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
    month = dt.month
    year = dt.year
    
    # Calculate 4-hour time bin (0-5)
    # 0: 00:00-03:59
    # 1: 04:00-07:59
    # 2: 08:00-11:59
    # 3: 12:00-15:59
    # 4: 16:00-19:59
    # 5: 20:00-23:59
    time_bin = hour // 4
    
    # Weekend indicator (Saturday=5, Sunday=6)
    is_weekend = 1 if day_of_week >= 5 else 0
    
    features = {
        'time_bin': time_bin,
        'day_of_week': day_of_week,
        'month': month,
        'year': year,
        'is_weekend': is_weekend
    }
    
    logger.debug(f"Extracted temporal features from {dt}: {features}")
    
    return features


def parse_datetime_string(datetime_str: str) -> datetime:
    """
    Parse datetime string to datetime object
    Supports multiple formats
    
    Args:
        datetime_str: DateTime string
        
    Returns:
        datetime object
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S",      # ISO format: 2026-02-20T14:30:00
        "%Y-%m-%d %H:%M:%S",      # Space separator: 2026-02-20 14:30:00
        "%Y-%m-%dT%H:%M:%S.%f",   # ISO with microseconds
        "%Y-%m-%d",               # Date only: 2026-02-20
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt)
        except ValueError:
            continue
    
    # If none of the formats work, raise error
    raise ValueError(
        f"Cannot parse datetime string: {datetime_str}. "
        f"Expected format: YYYY-MM-DDTHH:MM:SS"
    )


def get_time_bin_description(time_bin: int) -> str:
    """Get human-readable description of time bin"""
    descriptions = {
        0: "00:00-03:59 (Late Night)",
        1: "04:00-07:59 (Early Morning)",
        2: "08:00-11:59 (Morning)",
        3: "12:00-15:59 (Afternoon)",
        4: "16:00-19:59 (Evening)",
        5: "20:00-23:59 (Night)"
    }
    return descriptions.get(time_bin, "Unknown")


def validate_temporal_features(features: Dict[str, int]) -> bool:
    """
    Validate temporal features are within expected ranges
    
    Args:
        features: Dictionary with temporal features
        
    Returns:
        True if valid, False otherwise
    """
    try:
        time_bin = features.get('time_bin')
        day_of_week = features.get('day_of_week')
        month = features.get('month')
        year = features.get('year')
        is_weekend = features.get('is_weekend')
        
        # Validate ranges
        if not (0 <= time_bin <= 5):
            return False
        if not (0 <= day_of_week <= 6):
            return False
        if not (1 <= month <= 12):
            return False
        if not (2000 <= year <= 2100):
            return False
        if not (is_weekend in [0, 1]):
            return False
        
        return True
        
    except (TypeError, AttributeError):
        return False
