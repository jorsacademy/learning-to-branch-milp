"""Branching rules and strong-branching scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltb_milp.problem import BinaryPackingMILP, solve_lp_relaxation


@dataclass(frozen=True)
class BranchDecision:
    variable: int
    extra_lp_solves: int = 0


def fractional_candidates(x: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Return indices of variables that are fractional in an LP solution."""
    return np.flatnonzero((x > tol) & (x < 1.0 - tol))


def most_fractional(x: np.ndarray) -> BranchDecision:
    """Choose the candidate closest to 0.5."""
    candidates = fractional_candidates(x)
    if candidates.size == 0:
        raise ValueError("no fractional branching candidates")
    distances = np.abs(x[candidates] - 0.5)
    return BranchDecision(int(candidates[np.argmin(distances)]))


def strong_branch_scores(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    parent_objective: float,
    bounds: dict[int, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Score all fractional candidates by the minimum child bound improvement."""
    candidates = fractional_candidates(x)
    if candidates.size == 0:
        raise ValueError("no fractional branching candidates")

    scores = np.zeros(candidates.size, dtype=float)
    base_bounds = dict(bounds or {})
    solves = 0
    fallback_gain = abs(parent_objective) + 1.0

    for pos, variable in enumerate(candidates):
        child_objectives: list[float | None] = []
        for value in (0.0, 1.0):
            child_bounds = dict(base_bounds)
            child_bounds[int(variable)] = (value, value)
            child = solve_lp_relaxation(problem, child_bounds)
            solves += 1
            child_objectives.append(child.objective if child.feasible else None)

        gains = [
            fallback_gain if objective is None else max(0.0, parent_objective - objective)
            for objective in child_objectives
        ]
        scores[pos] = min(gains) + 1e-6 * max(gains)

    return candidates, scores, solves


def strong_branch(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    parent_objective: float,
    bounds: dict[int, tuple[float, float]] | None = None,
) -> BranchDecision:
    candidates, scores, solves = strong_branch_scores(problem, x, parent_objective, bounds)
    return BranchDecision(int(candidates[np.argmax(scores)]), extra_lp_solves=solves)
