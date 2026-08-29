import math

from ltb_milp.statistics import summarize_samples


def test_summarize_samples_computes_mean_std_and_ci() -> None:
    summary = summarize_samples([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(summary.mean, 2.5)
    assert math.isclose(summary.std, 1.2909944487358056)
    assert math.isclose(summary.ci95, 1.96 * summary.std / 2.0)
    assert summary.n == 4


def test_summarize_samples_single_value_has_zero_uncertainty() -> None:
    summary = summarize_samples([7.0])
    assert summary.mean == 7.0
    assert summary.std == 0.0
    assert summary.ci95 == 0.0
    assert summary.n == 1


def test_summarize_samples_rejects_empty_input() -> None:
    try:
        summarize_samples([])
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")
