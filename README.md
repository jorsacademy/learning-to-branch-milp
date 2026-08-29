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
- preserve one candidate group per B&B node,
- extract candidate-level features,
- train an MLP with score regression or listwise expert-choice ranking,
- compare learned branching against classical baselines with repeated-seed statistics,
- evaluate learned policies under size and packing-distribution shift.

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

Internally the LP relaxation is solved with `scipy.optimize.linprog` after converting the maximization objective to minimization form. The generator can also fix the RHS `tightness` as a fraction of each row's total coefficient mass, enabling controlled distribution-shift experiments.

## Branching policies

- **Most fractional:** branch on the variable closest to 0.5.
- **Strong branching:** temporarily solve both child LP relaxations for every candidate and choose the candidate with the strongest bound improvement.
- **Learned policy:** rank candidate variables from inexpensive node/variable features using an MLP trained on strong-branching labels.

## Imitation data

The default training dataset follows strong branching through multiple branch-and-bound nodes rather than sampling only root nodes. At every fractional LP node:

1. candidate variables are identified,
2. strong-branching scores are computed,
3. candidate features and normalized expert scores are stored as one candidate group,
4. the best strong-branching variable is selected,
5. the resulting child nodes are explored depth-first,
6. incumbent-based LP-bound pruning is applied.

This exposes the learned model to the distribution shift that occurs between the root relaxation and deeper search-tree states. The original root-only dataset remains available as a controlled baseline.

## Ranking objective

Two training objectives are available:

- `mse`: regress normalized strong-branching scores candidate by candidate.
- `listwise`: treat each B&B node as a candidate list and minimize cross-entropy on the strong-branching expert's top-ranked variable.

`listwise` is the default because branching is fundamentally a within-node ranking decision rather than an absolute score-prediction problem.

The imitation metric is **top-1 strong-branching agreement**: the fraction of candidate groups where the learned policy and the expert choose the same variable.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Train

Full-tree data and listwise ranking are the defaults:

```bash
python scripts/train.py \
  --instances 200 \
  --vars 12 \
  --constraints 5 \
  --max-nodes-per-instance 64 \
  --loss listwise \
  --epochs 200 \
  --checkpoint checkpoints/branching_listwise.pt
```

Train the score-regression baseline separately:

```bash
python scripts/train.py \
  --instances 200 \
  --vars 12 \
  --constraints 5 \
  --max-nodes-per-instance 64 \
  --loss mse \
  --epochs 200 \
  --checkpoint checkpoints/branching_mse.pt
```

Root-only data remains available with `--dataset root`.

## Statistical benchmark

The benchmark treats each seed as an independent replicate. Within a seed, every branching policy is evaluated on exactly the same generated MILP instances. Metrics are first averaged within each seed; mean, sample standard deviation, and a normal-approximation 95% confidence-interval half-width are then computed across seed means.

```bash
python scripts/benchmark.py \
  --instances-per-seed 20 \
  --vars 12 \
  --constraints 5 \
  --seeds 1000 1001 1002 1003 1004 \
  --mse-checkpoint checkpoints/branching_mse.pt \
  --listwise-checkpoint checkpoints/branching_listwise.pt
```

The comparison includes most-fractional, strong branching, learned MSE, and learned listwise policies, with node count, LP solves, wall-clock time, objective consistency checks, held-out expert agreement, and mean / sample std / 95% CI across seeds.

## Generalization benchmark

A separate OOD benchmark evaluates one learned checkpoint on problem distributions that differ from the nominal `12 variables / 5 constraints` training regime:

- in-distribution `12v / 5c`,
- larger `16v / 5c`,
- larger and more constrained `20v / 8c`,
- tighter packing with `tightness=0.30`,
- looser packing with `tightness=0.70`.

```bash
python scripts/generalization.py \
  --checkpoint checkpoints/branching_listwise.pt \
  --instances-per-seed 10 \
  --seeds 2000 2001 2002 2003 2004
```

Each scenario uses paired instances across most-fractional, strong, and learned branching and reports repeated-seed mean/std/95% CI for B&B node count, LP solves, and wall-clock time. This tests whether the candidate-feature MLP transfers beyond the distribution used to generate its imitation data.

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
│   ├── statistics.py
│   └── training.py
├── scripts/
│   ├── train.py
│   ├── benchmark.py
│   └── generalization.py
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

The current MLP remains a baseline. Planned stages are graph-based MILP representations and stronger classical branching baselines such as pseudocost and reliability branching.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
