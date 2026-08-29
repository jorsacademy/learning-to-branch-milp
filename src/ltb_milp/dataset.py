"""Imitation dataset generation from strong-branching scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ltb_milp.branching import strong_branch_scores
from ltb_milp.features import candidate_features
from ltb_milp.problem import generate_binary_packing, solve_lp_relaxation


@dataclass(frozen=True)
class BranchingDataset:
    features: np.ndarray
    targets: np.ndarray


def collect_root_strong_branching_dataset(
    n_instances: int,
    n_vars: int,
    n_constraints: int,
    *,
    seed: int = 0,
) -> BranchingDataset:
    """Collect candidate features and normalized strong-branching scores at root nodes."""
    if n_instances <= 0:
        raise ValueError("n_instances must be positive")

    feature_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for offset in range(n_instances):
        problem = generate_binary_packing(n_vars, n_constraints, seed=seed + offset)
        lp = solve_lp_relaxation(problem)
        if not lp.feasible or lp.x is None:
            continue
        candidates, features = candidate_features(problem, lp.x)
        if candidates.size == 0:
            continue
        score_candidates, scores, _ = strong_branch_scores(problem, lp.x, lp.objective)
        if not np.array_equal(candidates, score_candidates):
            raise RuntimeError("candidate ordering mismatch")
        scale = max(float(np.max(np.abs(scores))), 1e-8)
        feature_rows.append(features)
        targets.append(scores / scale)

    if not feature_rows:
        raise RuntimeError("no fractional root nodes were generated")
    return BranchingDataset(np.vstack(feature_rows), np.concatenate(targets))
