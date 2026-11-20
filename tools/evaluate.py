"""Compatibility command; reusable code lives in evaluation.runner."""

import sys

if __package__ == "tools":
    from milk_vision.evaluation import runner as _implementation
    from milk_vision.evaluation.runner import (
        EvaluationRow,
        Prediction,
        _parser,
        _predictor,
        _sha256,
        iter_results,
        main,
        metrics,
        select_cases,
        summarize,
    )
else:
    from ..milk_vision.evaluation import runner as _implementation
    from ..milk_vision.evaluation.runner import (
        EvaluationRow,
        Prediction,
        _parser,
        _predictor,
        _sha256,
        iter_results,
        main,
        metrics,
        select_cases,
        summarize,
    )
__all__ = [
    "EvaluationRow",
    "Prediction",
    "_parser",
    "_predictor",
    "_sha256",
    "iter_results",
    "main",
    "metrics",
    "select_cases",
    "summarize",
]
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
