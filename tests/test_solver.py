from __future__ import annotations

import itertools

import numpy as np

from ltb_milp.branching import fractional_candidates, strong_branch_scores
from ltb_milp.problem import BinaryPackingMILP, generate_binary_packing, solve_lp_relaxation
from ltb_milp.solver import solve_branch_and_bound


def brute_force_optimum(problem: BinaryPackingMILP) -> float:
    best = float("-inf")
    for bits in itertools.product((0.0, 1.0), repeat=problem.n_vars):
        x = np.asarray(bits)
        if np.all(problem.A @ x <= problem.b + 1e-9):
            best = max(best, float(problem.c @ x))
    return best


def test_lp_relaxation_upper_bounds_integer_optimum() -> None:
    problem = generate_binary_packing(7, 3, seed=4)
    lp = solve_lp_relaxation(problem)
    assert lp.feasible
    assert lp.objective + 1e-8 >= brute_force_optimum(problem)


def test_branch_and_bound_matches_brute_force() -> None:
    problem = generate_binary_packing(8, 3, seed=12)
    expected = brute_force_optimum(problem)
    for policy in ("most_fractional", "strong"):
        result = solve_branch_and_bound(problem, policy=policy)
        assert result.optimal
        assert result.objective == expected
        assert result.solution is not None
        assert np.all(problem.A @ result.solution <= problem.b + 1e-8)


def test_strong_branch_scores_cover_all_fractional_candidates() -> None:
    problem = generate_binary_packing(10, 4, seed=2)
    lp = solve_lp_relaxation(problem)
    assert lp.feasible and lp.x is not None
    candidates = fractional_candidates(lp.x)
    if candidates.size:
        scored, scores, solves = strong_branch_scores(problem, lp.x, lp.objective)
        assert np.array_equal(scored, candidates)
        assert scores.shape == candidates.shape
        assert solves == 2 * candidates.size
        assert np.all(scores >= 0)
