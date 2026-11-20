"""Compatibility command; reusable code lives in evaluation.dataset."""

import sys

if __package__ == "tools":
    from milk_vision.evaluation import dataset as _implementation
    from milk_vision.evaluation.dataset import (
        DEFAULT_ROOT,
        NET_FOLDERS,
        VideoCase,
        _pairs,
        _video_path,
        main,
        read_annotations,
    )
else:
    from ..milk_vision.evaluation import dataset as _implementation
    from ..milk_vision.evaluation.dataset import (
        DEFAULT_ROOT,
        NET_FOLDERS,
        VideoCase,
        _pairs,
        _video_path,
        main,
        read_annotations,
    )
__all__ = [
    "DEFAULT_ROOT",
    "NET_FOLDERS",
    "VideoCase",
    "_pairs",
    "_video_path",
    "main",
    "read_annotations",
]
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
