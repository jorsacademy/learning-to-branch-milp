"""Statistical summaries for repeated-seed MILP benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics


@dataclass(frozen=True)
class Summary:
    """Mean, sample standard deviation, and normal-approximation 95% CI half-width."""

    mean: float
    std: float
    ci95: float
    n: int


def summarize_samples(values: list[float]) -> Summary:
    """Summarize independent replicate values.

    The 95% interval uses 1.96 * standard_error. For small replicate counts this is
    an approximation; the benchmark reports the replicate count explicitly.
    """
    if not values:
        raise ValueError("values must not be empty")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return Summary(mean=mean, std=std, ci95=ci95, n=len(values))
