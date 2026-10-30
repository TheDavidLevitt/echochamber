"""Bootstrap intervals and multiple-comparison correction on synthetic data."""

import random

import pytest

from echochamber.core.stats import (
    bootstrap_cluster_mean,
    bootstrap_mean_diff,
    holm_adjust,
    percentile,
)


def noisy(mean, n=20, sd=25.0, seed=1):
    """n judge-like margins around `mean`, clipped to the -100..100 scale."""
    rng = random.Random(seed)
    return [max(-100.0, min(100.0, rng.gauss(mean, sd))) for _ in range(n)]


def test_strong_effect_is_detected():
    ablated, baseline = noisy(40, seed=1), noisy(-10, seed=2)
    result = bootstrap_mean_diff(ablated, baseline, n_resamples=4000, seed=0)
    assert result.estimate > 30
    assert result.excludes_null()
    assert result.ci_low > 0
    assert result.p_value < 0.01


def test_null_effect_is_not_detected():
    # Same underlying distribution, different draws
    a, b = noisy(10, seed=3), noisy(10, seed=4)
    result = bootstrap_mean_diff(a, b, n_resamples=4000, seed=0)
    assert result.ci_low < 0 < result.ci_high
    assert not result.excludes_null()
    assert result.p_value > 0.05


def test_identical_groups_give_p_of_one():
    values = [10.0, 20.0, 30.0, 40.0]
    result = bootstrap_mean_diff(values, list(values), n_resamples=2000, seed=0)
    assert result.estimate == 0.0
    assert not result.excludes_null()
    assert result.p_value > 0.5


def test_fixed_seed_is_deterministic():
    a, b = noisy(15, seed=5), noisy(0, seed=6)
    first = bootstrap_mean_diff(a, b, n_resamples=2000, seed="baseline:minus x")
    second = bootstrap_mean_diff(a, b, n_resamples=2000, seed="baseline:minus x")
    assert first == second
    other = bootstrap_mean_diff(a, b, n_resamples=2000, seed="another seed")
    assert other.estimate == first.estimate  # the point estimate never depends on the seed
    assert (other.ci_low, other.ci_high) != (first.ci_low, first.ci_high)


def test_wider_confidence_gives_wider_interval():
    a, b = noisy(15, seed=7), noisy(0, seed=8)
    narrow = bootstrap_mean_diff(a, b, n_resamples=3000, confidence=0.80, seed=0)
    wide = bootstrap_mean_diff(a, b, n_resamples=3000, confidence=0.99, seed=0)
    assert wide.ci_low < narrow.ci_low
    assert wide.ci_high > narrow.ci_high


def test_too_few_values_is_an_error():
    with pytest.raises(ValueError):
        bootstrap_mean_diff([1.0], [1.0, 2.0])


def test_cluster_bootstrap_detects_consistent_wins():
    # Candidate wins both sides of 5 of 6 topics, splits the last
    clusters = [[1.0, 1.0]] * 5 + [[1.0, 0.0]]
    result = bootstrap_cluster_mean(clusters, null=0.5, n_resamples=4000, seed=0)
    assert result.estimate == pytest.approx(11 / 12)
    assert result.ci_low > 0.5
    assert result.excludes_null()


def test_cluster_bootstrap_side_bias_is_not_an_effect():
    # A judge that always favors the prosecution: 1 win + 1 loss per topic
    clusters = [[1.0, 0.0]] * 4
    result = bootstrap_cluster_mean(clusters, null=0.5, n_resamples=2000, seed=0)
    assert result.estimate == 0.5
    # Every resample is exactly 0.5 because pairs are kept together
    assert result.ci_low == result.ci_high == 0.5
    assert not result.excludes_null()


def test_holm_textbook_example():
    # Worked example: sorted p = .005, .01, .03, .04 with m = 4
    #   .005*4 = .02, .01*3 = .03, .03*2 = .06, .04*1 = .04 -> monotone .06
    raw = [0.01, 0.04, 0.03, 0.005]
    assert holm_adjust(raw) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_handles_empty():
    assert holm_adjust([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([]) == []
    assert holm_adjust([0.02]) == [0.02]


def test_percentile_matches_linear_interpolation():
    data = [1.0, 2.0, 3.0, 4.0]
    assert percentile(data, 0.0) == 1.0
    assert percentile(data, 1.0) == 4.0
    assert percentile(data, 0.5) == 2.5
    assert percentile(data, 0.25) == pytest.approx(1.75)
