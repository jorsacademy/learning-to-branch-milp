# Learning to Branch in MILP

A research-oriented implementation of **machine-learning-guided branching for mixed-integer linear programming (MILP)**.

> **License:** source-available for non-commercial use only. See `LICENSE`.

## Motivation

Branch-and-bound performance depends heavily on the branching variable selected at each fractional LP relaxation. Classical solvers use rules such as most-infeasible branching, pseudocosts, reliability branching, and strong branching. Learning-to-branch methods treat strong branching as an expensive expert and train a cheaper policy to imitate its decisions.

This repository builds that pipeline from first principles on small binary MILPs:

- solve LP relaxations,
- identify fractional branching candidates,
- score candidates with strong branching,
- extract candidate-level features,
- collect imitation targets from the expert,
- train an MLP to predict strong-branching scores,
- compare learned branching against classical baselines using node counts and LP solve counts.

The implementation is intentionally small and transparent. It is not a replacement for production solvers such as SCIP, Gurobi, or CPLEX.

## Initial problem family

The first benchmark uses random binary packing MILPs:

\[
\max c^T x
\]

subject to

\[
Ax \le b, \qquad x \in \{0,1\}^n.
\]

Internally the LP relaxation is solved with `scipy.optimize.linprog` after converting the maximization objective to minimization form.

## Branching policies

- **Most fractional:** branch on the variable closest to 0.5.
- **Strong branching:** temporarily solve both child LP relaxations for every fractional candidate and choose the candidate with the strongest lower child-bound improvement.
- **Learned policy:** rank candidate variables from inexpensive features using an MLP trained to regress normalized strong-branching scores.

The first learned dataset uses root-node candidates. This makes the imitation pipeline reproducible and easy to audit before extending data collection to deeper branch-and-bound nodes.

## Candidate features

The initial MLP receives six features per fractional candidate:

- LP value,
- fractionality,
- normalized objective coefficient,
- normalized column activity,
- constraint-column density,
- normalized coefficient magnitude summary.

## Project structure

```text
.
├── src/ltb_milp/
│   ├── problem.py
│   ├── branching.py
│   ├── features.py
│   ├── solver.py
│   ├── dataset.py
│   ├── models.py
│   ├── training.py
│   └── policies.py
├── scripts/
│   ├── train.py
│   └── benchmark.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Train the branching model

Generate strong-branching supervision and fit the MLP:

```bash
python scripts/train.py \
  --instances 200 \
  --vars 12 \
  --constraints 5 \
  --epochs 200 \
  --checkpoint checkpoints/branching_mlp.pt
```

## Benchmark branching policies

Compare most-fractional and strong branching:

```bash
python scripts/benchmark.py --instances 25 --vars 12 --constraints 5
```

Include the learned policy:

```bash
python scripts/benchmark.py \
  --instances 25 \
  --vars 12 \
  --constraints 5 \
  --checkpoint checkpoints/branching_mlp.pt
```

The benchmark verifies that all policies recover the same optimum and reports mean/std for:

- branch-and-bound nodes processed,
- total LP relaxations solved,
- final objective value.

Strong branching typically spends more LP solves per node because it evaluates candidate children explicitly. The learned policy is intended to approximate strong branching without paying those extra expert LP solves at inference time.

## Tests

The test suite includes a brute-force cross-check on small instances to verify that branch-and-bound returns the exact binary optimum.

```bash
pytest
ruff check .
```

## Research roadmap

Planned extensions include:

- collect imitation data at deeper branch-and-bound nodes,
- pseudocost and reliability-branching baselines,
- candidate ranking losses instead of pure score regression,
- bipartite MILP graph representations,
- GNN branching policies following the Gasse et al. line,
- harder instance families such as set covering and combinatorial auctions,
- repeated-seed confidence intervals and branching-agreement statistics.

## Research lineage

This repository follows the learning-to-branch line associated with Khalil et al. and Gasse et al., while keeping the first implementation solver-independent and educational.

## References

1. Khalil, E. B. et al. (2016). Learning to Branch in Mixed Integer Programming. AAAI.
2. Gasse, M. et al. (2019). Exact Combinatorial Optimization with Graph Convolutional Neural Networks. NeurIPS.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
