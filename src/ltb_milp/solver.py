"""Branch-and-bound solver used to benchmark branching policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ltb_milp.branching import BranchDecision, most_fractional, strong_branch
from ltb_milp.problem import BinaryPackingMILP, solve_lp_relaxation

BranchPolicy = Callable[
    [BinaryPackingMILP, np.ndarray, float, dict[int, tuple[float, float]]], BranchDecision
]


@dataclass(frozen=True)
class SolveResult:
    objective: float
    solution: np.ndarray | None
    nodes_processed: int
    lp_solves: int
    optimal: bool


def _most_fractional_policy(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    objective: float,
    bounds: dict[int, tuple[float, float]],
) -> BranchDecision:
    del problem, objective, bounds
    return most_fractional(x)


def _strong_policy(
    problem: BinaryPackingMILP,
    x: np.ndarray,
    objective: float,
    bounds: dict[int, tuple[float, float]],
) -> BranchDecision:
    return strong_branch(problem, x, objective, bounds)


def solve_branch_and_bound(
    problem: BinaryPackingMILP,
    *,
    policy: BranchPolicy | str = "most_fractional",
    max_nodes: int = 10_000,
    integrality_tol: float = 1e-7,
) -> SolveResult:
    """Solve a small binary MILP with depth-first branch-and-bound."""
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")
    if isinstance(policy, str):
        if policy == "most_fractional":
            branch_policy = _most_fractional_policy
        elif policy == "strong":
            branch_policy = _strong_policy
        else:
            raise ValueError("unknown branching policy")
    else:
        branch_policy = policy

    incumbent = float("-inf")
    incumbent_x: np.ndarray | None = None
    stack: list[dict[int, tuple[float, float]]] = [{}]
    nodes = 0
    lp_solves = 0

    while stack and nodes < max_nodes:
        bounds = stack.pop()
        lp = solve_lp_relaxation(problem, bounds)
        nodes += 1
        lp_solves += 1
        if not lp.feasible or lp.objective <= incumbent + 1e-9:
            continue
        assert lp.x is not None
        rounded = np.rint(lp.x)
        if np.all(np.abs(lp.x - rounded) <= integrality_tol):
            value = float(problem.c @ rounded)
            if value > incumbent:
                incumbent = value
                incumbent_x = rounded.astype(float)
            continue

        decision = branch_policy(problem, lp.x, lp.objective, bounds)
        lp_solves += decision.extra_lp_solves
        variable = decision.variable
        down = dict(bounds)
        up = dict(bounds)
        down[variable] = (0.0, 0.0)
        up[variable] = (1.0, 1.0)
        # Push the higher-value branch last so it is explored first under LIFO.
        if lp.x[variable] >= 0.5:
            stack.extend([down, up])
        else:
            stack.extend([up, down])

    return SolveResult(
        objective=incumbent,
        solution=incumbent_x,
        nodes_processed=nodes,
        lp_solves=lp_solves,
        optimal=not stack,
    )
