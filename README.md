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
- train an MLP with score regression or listwise expert-choice ranking,
- represent MILPs as bipartite variable-constraint graphs,
- train a pure-PyTorch message-passing GNN branching policy,
- compare learned policies against most-fractional, pseudocost, and strong branching,
- evaluate policies with repeated-seed statistics and distribution-shift benchmarks.

The implementation is intentionally small and transparent. It is not a replacement for production solvers such as SCIP, Gurobi, or CPLEX.

## Initial problem family

The benchmark uses random binary packing MILPs:

\[
\max c^T x
\]

subject to

\[
Ax \le b, \qquad x \in \{0,1\}^n.
\]

Internally the LP relaxation is solved with `scipy.optimize.linprog`. The generator also supports controlled RHS `tightness` for distribution-shift experiments.

## Branching policies

- **Most fractional:** branch on the variable closest to 0.5.
- **Pseudocost:** maintain historical per-variable down/up unit bound degradations from solved child LPs, then score candidates using the predicted two-sided degradation.
- **Strong branching:** solve both child LP relaxations for each candidate and choose the strongest bound improvement.
- **MLP learned policy:** rank candidates from inexpensive hand-engineered features.
- **Bipartite GNN policy:** score candidates after variable-to-constraint and constraint-to-variable message passing over the MILP coefficient graph.

Pseudocost branching starts with neutral fallback estimates and updates its statistics online as the search observes child-node LP bounds. It therefore avoids the repeated probing LP solves required by strong branching.

## Imitation data and ranking

Full-tree imitation data follows strong branching through multiple B&B nodes. Candidate features and normalized strong-branching scores are stored per node. Two MLP objectives are available:

- `mse`: regress normalized strong-branching scores.
- `listwise`: cross-entropy over the expert's top-ranked candidate at each node.

Top-1 strong-branching agreement is used as the imitation metric.

## Bipartite GNN representation

The graph model uses:

- one node for each variable,
- one node for each constraint,
- normalized MILP coefficients as bipartite edge weights,
- variable features including LP value, fractionality, objective coefficient, and column density,
- constraint features including RHS, LP activity, slack, and row density,
- two-way message passing before candidate scoring.

The GNN is implemented in pure PyTorch, without PyTorch Geometric or DGL. Inference graph construction does **not** call strong branching; expert calls are used only when generating labeled training states.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Train MLP

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

MSE baseline:

```bash
python scripts/train.py --loss mse --checkpoint checkpoints/branching_mse.pt
```

## Train bipartite GNN

```bash
python scripts/train_gnn.py \
  --instances 100 \
  --vars 12 \
  --constraints 5 \
  --max-nodes-per-instance 32 \
  --epochs 100 \
  --hidden-dim 64 \
  --checkpoint checkpoints/branching_gnn.pt
```

The GNN is trained with listwise expert-choice cross entropy and reports training-set top-1 expert agreement.

## Statistical benchmark

The repeated-seed benchmark evaluates active learned models and classical policies on exactly the same MILP instances:

```bash
python scripts/benchmark.py \
  --instances-per-seed 20 \
  --vars 12 \
  --constraints 5 \
  --seeds 1000 1001 1002 1003 1004 \
  --mse-checkpoint checkpoints/branching_mse.pt \
  --listwise-checkpoint checkpoints/branching_listwise.pt \
  --gnn-checkpoint checkpoints/branching_gnn.pt
```

The comparison includes:

- most-fractional branching,
- pseudocost branching,
- strong branching,
- learned MLP score regression,
- learned MLP listwise ranking,
- learned bipartite GNN,
- branch-and-bound node count,
- LP solve count,
- wall-clock time,
- objective-consistency checks,
- held-out strong-branching top-1 agreement for learned models,
- mean / sample std / normal-approximation 95% CI across seeds.

## Generalization benchmark

MLP and GNN checkpoints can be evaluated together under distribution shift:

```bash
python scripts/generalization.py \
  --checkpoint checkpoints/branching_listwise.pt \
  --gnn-checkpoint checkpoints/branching_gnn.pt \
  --instances-per-seed 10 \
  --seeds 2000 2001 2002 2003 2004
```

Scenarios include the nominal `12v/5c` distribution, larger `16v/5c`, `20v/8c`, tighter packing, and looser packing. Most-fractional, pseudocost, strong, and active learned policies solve the same generated instances before repeated-seed summaries are computed.

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
│   ├── graph.py
│   ├── graph_dataset.py
│   ├── gnn.py
│   ├── policies.py
│   ├── statistics.py
│   └── training.py
├── scripts/
│   ├── train.py
│   ├── train_gnn.py
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

The repository follows the learning-to-branch line associated with Khalil et al. and Gasse et al. The MLP remains a transparent baseline, the bipartite message-passing model provides a graph-based learned policy, and pseudocost now provides a stronger low-overhead classical baseline. The next classical stage is reliability branching, which combines pseudocost history with selective strong-branch probing for insufficiently observed candidates.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
