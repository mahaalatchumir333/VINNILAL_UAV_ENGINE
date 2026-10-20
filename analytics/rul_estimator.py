"""
rul_estimator.py
-------------------
Prototype degradation-index-based life estimate under simulated operating
assumptions. This is explicitly NOT validated aerospace RUL: no
run-to-failure data exists for this prototype. Every output must carry
an assumption statement and a confidence qualifier.

Includes a numeric uncertainty interval (not just a confidence label),
derived from the regression's own statistical uncertainty in the fitted
degradation slope. This is still a PROTOTYPE interval -- it reflects how
noisy the recent trend is, not a validated prognostic confidence bound.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
from scipy import stats


FAILURE_HEALTH_THRESHOLD = 30.0  # health index at which "failure" is defined
Z_90 = 1.645  # one-sided z-score for a 90% confidence bound


@dataclass
class RULResult:
    estimated_minutes: Optional[float]
    lower_bound_minutes: Optional[float]   # optimistic (fails sooner is impossible; this is the SHORTER estimate)
    upper_bound_minutes: Optional[float]   # pessimistic (slower degradation case)
    confidence: str
    assumption: str
    degradation_rate_per_min: float
    slope_std_error: Optional[float] = None


class RULEstimator:
    """
    Maintains a short rolling history of the overall health index and
    fits a linear trend to project time-to-threshold, propagating the
    regression's slope uncertainty into a [lower, upper] minutes range.
    """

    def __init__(self, window_minutes: int = 10, sample_interval_s: float = 1.0):
        self.max_samples = int(window_minutes * 60 / sample_interval_s)
        self.timestamps_s: Deque[float] = deque(maxlen=self.max_samples)
        self.health_values: Deque[float] = deque(maxlen=self.max_samples)

    def _propagate_uncertainty(self, current_health: float, slope: float,
                                slope_stderr: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Given the estimated slope +/- its standard error, compute the
        range of minutes-to-threshold implied by a faster-than-expected
        slope (lower bound on time remaining) and a slower-than-expected
        slope (upper bound on time remaining). Uses a 90% one-sided
        interval on the slope itself (Z_90), which is a simple prototype
        approximation, not a rigorous prognostic confidence bound.
        """
        if slope >= 0:
            return None, None
        if not np.isfinite(slope_stderr) or slope_stderr < 0:
            return None, None

        if slope_stderr == 0.0:
            # Perfect fit, no regression uncertainty -- bounds collapse to
            # the point estimate itself, not "unknown".
            point = (current_health - FAILURE_HEALTH_THRESHOLD) / abs(slope)
            point = round(max(0.0, point), 1)
            return point, point

        worse_slope = slope - Z_90 * slope_stderr   # more negative -> faster failure
        better_slope = slope + Z_90 * slope_stderr  # less negative (or positive) -> slower/no failure

        lower_minutes = (current_health - FAILURE_HEALTH_THRESHOLD) / abs(worse_slope) \
            if worse_slope < 0 else 0.0

        upper_minutes = (current_health - FAILURE_HEALTH_THRESHOLD) / abs(better_slope) \
            if better_slope < 0 else None  # slope could flip non-degrading -> no finite upper bound

        return (
            round(max(0.0, lower_minutes), 1) if lower_minutes is not None else None,
            round(upper_minutes, 1) if upper_minutes is not None else None,
        )

    def update(self, t_seconds: float, health_index: float) -> RULResult:
        self.timestamps_s.append(t_seconds)
        self.health_values.append(health_index)

        if len(self.health_values) < 10:
            return RULResult(
                estimated_minutes=None, lower_bound_minutes=None, upper_bound_minutes=None,
                confidence="insufficient_data",
                assumption="Not enough history to fit a degradation trend yet.",
                degradation_rate_per_min=0.0,
            )

        x = np.array(self.timestamps_s, dtype=float)
        x = (x - x[0]) / 60.0  # minutes since window start
        y = np.array(self.health_values, dtype=float)

        try:
            lin = stats.linregress(x, y)
            slope, intercept = lin.slope, lin.intercept
            slope_stderr = lin.stderr
        except Exception:
            slope, intercept, slope_stderr = 0.0, y[-1], float("nan")

        current_health = y[-1]

        if slope >= -0.01:
            return RULResult(
                estimated_minutes=None, lower_bound_minutes=None, upper_bound_minutes=None,
                confidence="not_applicable",
                assumption="No sustained degradation trend detected.",
                degradation_rate_per_min=round(float(slope), 4),
                slope_std_error=round(float(slope_stderr), 4) if np.isfinite(slope_stderr) else None,
            )

        minutes_to_threshold = (current_health - FAILURE_HEALTH_THRESHOLD) / abs(slope)
        minutes_to_threshold = max(0.0, minutes_to_threshold)

        lower, upper = self._propagate_uncertainty(current_health, slope, slope_stderr)

        # confidence heuristic: combine fit residual error AND relative width
        # of the uncertainty interval (a wide interval means "medium" is a lie)
        residuals = y - (slope * x + intercept)
        fit_error = float(np.std(residuals))
        interval_width = (upper - lower) if (upper is not None and lower is not None) else None

        if fit_error < 2.0 and interval_width is not None and interval_width < 0.5 * minutes_to_threshold:
            confidence = "medium"
        elif fit_error < 5.0:
            confidence = "low"
        else:
            confidence = "very_low"

        return RULResult(
            estimated_minutes=round(minutes_to_threshold, 1),
            lower_bound_minutes=lower,
            upper_bound_minutes=upper,
            confidence=confidence,
            assumption=(
                "Scenario-based estimate assuming the current degradation "
                "rate continues unchanged. The [lower, upper] range reflects "
                "statistical uncertainty in the fitted trend's slope only -- "
                "it does NOT account for unmodeled failure modes, sudden "
                "faults, or model error, and is not equivalent to validated "
                "run-to-failure aerospace RUL."
            ),
            degradation_rate_per_min=round(float(slope), 4),
            slope_std_error=round(float(slope_stderr), 4) if np.isfinite(slope_stderr) else None,
        )
