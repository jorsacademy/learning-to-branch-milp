# Learning to Branch in MILP

A research-oriented implementation of **machine-learning-guided branching for mixed-integer linear programming (MILP)**.

> **License:** source-available for non-commercial use only. See `LICENSE`.

## Motivation

Branch-and-bound performance depends heavily on the branching variable selected at each fractional LP relaxation. Classical solvers use rules such as most-infeasible branching, pseudocosts, reliability branching, and strong branching. Learning-to-branch methods treat strong branching as an expensive expert and train a cheaper policy to imitate its decisions.

This repository builds that pipeline from first principles on small binary MILPs:

- solve LP relaxations,
- identify fractional branching candidates,
- score candidates with strong branching,
- collect expert labels across complete search-tree trajectories,
- extract candidate-level features,
- train an MLP ranking policy by imitation learning,
- compare learned branching against classical baselines using node counts and solve statistics.

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
- **Strong branching:** temporarily solve both child LP relaxations for every candidate and choose the candidate with the strongest bound improvement.
- **Learned policy:** rank candidate variables from inexpensive node/variable features using an MLP trained on strong-branching labels.

## Imitation data

The default training dataset now follows strong branching through multiple branch-and-bound nodes rather than sampling only root nodes. At every fractional LP node:

1. candidate variables are identified,
2. strong-branching scores are computed,
3. candidate features and normalized expert scores are stored,
4. the best strong-branching variable is selected,
5. the resulting child nodes are explored depth-first,
6. incumbent-based LP-bound pruning is applied.

This exposes the learned model to the distribution shift that occurs between the root relaxation and deeper search-tree states.

The original root-only dataset remains available as a controlled baseline.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Train

Full-tree imitation data is the default:

```bash
python scripts/train.py \
  --instances 200 \
  --vars 12 \
  --constraints 5 \
  --max-nodes-per-instance 64 \
  --epochs 200 \
  --checkpoint checkpoints/branching_mlp.pt
```

For the root-only baseline:

```bash
python scripts/train.py --dataset root --instances 200 --vars 12 --constraints 5
```

## Benchmark

```bash
python scripts/benchmark.py \
  --instances 25 \
  --vars 12 \
  --constraints 5 \
  --checkpoint checkpoints/branching_mlp.pt
```

The benchmark compares most-fractional, strong, and learned branching while checking that all completed solves return the same optimum.

## Evaluation

The main solver metrics are:

- branch-and-bound nodes processed,
- LP relaxations solved,
- incumbent objective,
- optimality status,
- branching agreement with strong branching,
- repeated-seed mean/std summaries.

## Project structure

```text
.
├── src/ltb_milp/
│   ├── problem.py
│   ├── branching.py
│   ├── solver.py
│   ├── features.py
│   ├── dataset.py
│   ├── models.py
│   ├── policies.py
│   └── training.py
├── scripts/
│   ├── train.py
│   └── benchmark.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
└── LICENSE
```

## Tests

```bash
pytest
ruff check .
```

## Research lineage

This repository follows the learning-to-branch line associated with Khalil et al. and Gasse et al., while keeping the first implementation solver-independent and educational.

The current MLP is deliberately a baseline. Planned stages are ranking-oriented training objectives, statistical/generalization benchmarks, graph-based MILP representations, and stronger classical branching baselines.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
