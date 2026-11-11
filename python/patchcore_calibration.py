"""Place a calibrated boundary between validation scores, preserving decisions."""
import numpy as np


def stable_threshold(scores, threshold):
    """Keep the selected F1 partition without putting a sample on its boundary.

    A score-valued threshold can flip under tiny FP32 runtime differences. Only
    validation scores are used here. This is not a guarantee for future inputs.
    """
    values = np.asarray(scores, dtype=np.float64)
    boundary = float(threshold)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or not np.isfinite(boundary):
        raise ValueError("PatchCore 임계값 보정 점수 오류")
    below, above = values[values < boundary], values[values >= boundary]
    result = {"optimal_threshold": boundary, "threshold_selection": "validation_score_boundary",
              "threshold_margin": 0.0}
    if not below.size or not above.size:
        return result
    lower, upper = float(below.max()), float(above.min())
    # Native SDK thresholds use float32; require a representable interior value.
    midpoint = float(np.float32(lower / 2 + upper / 2))
    if lower < midpoint < upper and midpoint > 0:
        assert np.array_equal(values >= boundary, values >= midpoint)
        result.update(optimal_threshold=midpoint, threshold_selection="validation_score_midpoint",
                      threshold_margin=min(midpoint-lower, upper-midpoint))
    return result
