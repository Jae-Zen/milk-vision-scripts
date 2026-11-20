"""Stable project paths and shared checkpoint validation."""

from collections.abc import Mapping
from pathlib import Path

from .errors import VerificationError

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
PLACEMENT_MODEL = PROJECT_ROOT / "Models" / "milk-vision-v4s.pt"
MOTION_MODEL = PROJECT_ROOT / "Models" / "milk-vision-v4n.pt"
TRACKER_CONFIG = Path(__file__).parent / "observations" / "milk_botsort.yaml"


def resolve_milk_class_id(names: Mapping[int, str]) -> int:
    """Require only a milk class; other checkpoint classes are unused."""
    # YOLO.names normalizes checkpoint class names to an {id: name} dictionary.
    normalized = {
        str(class_name).strip().casefold(): int(class_id)
        for class_id, class_name in names.items()
    }

    if "milk" not in normalized:
        raise VerificationError("Model must contain a class named 'milk'.")
    return normalized["milk"]
