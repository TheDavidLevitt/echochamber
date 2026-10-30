"""Small, dependency-free statistics for ablation and eval reports.

Debate verdicts are noisy, so a raw difference in mean margin between two
conditions says little on its own. This module turns those differences into
claims with error bars:

- Percentile bootstrap confidence intervals: resample the observed runs
  with replacement many times, recompute the statistic each time, and read
  the interval off the middle 95% of the resampled values. No normality
  assumption, which suits bounded, lumpy judge margins.
- A bootstrap p-value obtained by inverting that interval (the smallest
  two-sided level at which the interval would exclude the null value).
- Holm-Bonferroni adjustment, for when one report makes many comparisons
  at once (one per ablated document).

Everything is seeded so the same records always produce the same report.
Pure standard library: the core package keeps zero hard dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, Union

Seed = Union[int, str]

DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with its percentile-bootstrap interval."""

    estimate: float
    ci_low: float
    ci_high: float
    p_value: float  # two-sided, against `null`
    null: float
    confidence: float
    n_resamples: int

    def excludes_null(self) -> bool:
        """True when the whole interval sits on one side of the null value."""
        return self.ci_low > self.null or self.ci_high < self.null


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """q-th quantile (0..1) of pre-sorted data, linear interpolation.

    Matches numpy's default ("linear") method.
    """
    if not sorted_values:
        raise ValueError("percentile of empty data")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _summarize(
    estimate: float,
    resampled: list[float],
    null: float,
    confidence: float,
) -> BootstrapResult:
    resampled.sort()
    b = len(resampled)
    alpha = 1.0 - confidence
    # Two-sided p-value by inverting the percentile interval. The +1 terms
    # count the observed data as one of the resamples, so p is never
    # reported as exactly 0 from a finite number of draws.
    at_or_below = sum(1 for v in resampled if v <= null)
    at_or_above = sum(1 for v in resampled if v >= null)
    p = min(1.0, 2.0 * (min(at_or_below, at_or_above) + 1) / (b + 1))
    return BootstrapResult(
        estimate=estimate,
        ci_low=percentile(resampled, alpha / 2),
        ci_high=percentile(resampled, 1 - alpha / 2),
        p_value=p,
        null=null,
        confidence=confidence,
        n_resamples=b,
    )


def bootstrap_mean_diff(
    sample: Sequence[float],
    reference: Sequence[float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: Seed = 0,
) -> BootstrapResult:
    """Bootstrap the difference mean(sample) - mean(reference).

    The two groups are independent runs (e.g. an ablated condition vs the
    baseline), so each is resampled separately. Needs at least two values
    per group: a single run carries no information about its own spread.
    """
    if len(sample) < 2 or len(reference) < 2:
        raise ValueError("need at least 2 values in each group to bootstrap")
    rng = random.Random(seed)
    n_s, n_r = len(sample), len(reference)
    resampled = [
        _mean(rng.choices(sample, k=n_s)) - _mean(rng.choices(reference, k=n_r))
        for _ in range(n_resamples)
    ]
    estimate = _mean(sample) - _mean(reference)
    return _summarize(estimate, resampled, 0.0, confidence)


def bootstrap_cluster_mean(
    clusters: Sequence[Sequence[float]],
    *,
    null: float = 0.0,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: Seed = 0,
) -> BootstrapResult:
    """Bootstrap the pooled mean of clustered observations.

    Whole clusters are resampled, not individual values, because values in a
    cluster are not independent. In the eval harness a cluster is one topic
    debated twice with sides swapped: the pair shares the topic's difficulty
    and the side bias that swapping is meant to cancel, so it must stay
    together when resampling.
    """
    clusters = [list(c) for c in clusters if c]
    if len(clusters) < 2:
        raise ValueError("need at least 2 non-empty clusters to bootstrap")
    rng = random.Random(seed)
    k = len(clusters)

    def pooled_mean(chosen: Sequence[Sequence[float]]) -> float:
        total = sum(sum(c) for c in chosen)
        count = sum(len(c) for c in chosen)
        return total / count

    resampled = [pooled_mean(rng.choices(clusters, k=k)) for _ in range(n_resamples)]
    return _summarize(pooled_mean(clusters), resampled, null, confidence)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order.

    Sort the m p-values ascending; the i-th smallest (0-based) is multiplied
    by (m - i), then a running maximum keeps the adjusted values monotone,
    and everything is capped at 1. Comparing adjusted values against alpha
    controls the family-wise error rate: the chance of even one false
    "this document matters" claim across the whole report stays <= alpha.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted
