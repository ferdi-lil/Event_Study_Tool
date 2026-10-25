"""Event-Study-Tool: configurable short-horizon event studies (MacKinlay 1997)."""
from .config import Event, Group, OutlierSettings, PlaceboSettings, StudyConfig
from .engine import StudyResult, run_study

__all__ = ["Event", "Group", "OutlierSettings", "PlaceboSettings", "StudyConfig", "StudyResult", "run_study"]
