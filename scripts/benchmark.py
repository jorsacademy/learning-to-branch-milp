from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from ltb_milp.models import BranchingMLP
from ltb_milp.policies import learned_branch_policy
from ltb_milp.problem import generate_binary_packing
from ltb_milp.solver import solve_branch_and_bound


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MILP branching policies.")
    parser.add_argument("--instances", type=int, default=25)
    parser.add_argument("--vars", type=int, default=12)
    parser.add_argument("--constraints", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def summarize(values: list[float]) -> str:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"mean={mean:.3f} std={std:.3f}"


def main() -> None:
    args = parse_args()
    policies: dict[str, object] = {
        "most_fractional": "most_fractional",
        "strong": "strong",
    }
    if args.checkpoint:
        model = BranchingMLP()
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model_state_dict"])
        policies["learned"] = learned_branch_policy(model)

    node_counts = {name: [] for name in policies}
    lp_counts = {name: [] for name in policies}
    objectives = {name: [] for name in policies}

    for offset in range(args.instances):
        problem = generate_binary_packing(
            args.vars,
            args.constraints,
            seed=args.seed + offset,
        )
        reference_objective: float | None = None
        for name, policy in policies.items():
            result = solve_branch_and_bound(problem, policy=policy)
            node_counts[name].append(float(result.nodes_processed))
            lp_counts[name].append(float(result.lp_solves))
            objectives[name].append(result.objective)
            if reference_objective is None:
                reference_objective = result.objective
            elif abs(result.objective - reference_objective) > 1e-6:
                raise RuntimeError("branching policies returned inconsistent optima")

    for name in policies:
        print(f"[{name}]")
        print(f"nodes: {summarize(node_counts[name])}")
        print(f"lp_solves: {summarize(lp_counts[name])}")
        print(f"objective: {summarize(objectives[name])}")


if __name__ == "__main__":
    main()
