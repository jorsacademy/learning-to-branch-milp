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
- train MLP score-regression and listwise-ranking policies,
- represent MILPs as bipartite variable-constraint graphs,
- train a pure-PyTorch message-passing GNN branching policy,
- compare learned policies with most-fractional, pseudocost, reliability, and strong branching,
- evaluate policies with repeated-seed, paired-comparison, and distribution-shift protocols.

The implementation is intentionally small and transparent. It is not a replacement for production solvers such as SCIP, Gurobi, or CPLEX.

## Problem family

The benchmark uses random binary packing MILPs:

\[
\max c^T x
\]

subject to

\[
Ax \le b, \qquad x \in \{0,1\}^n.
\]

LP relaxations are solved with `scipy.optimize.linprog`. The generator also supports controlled RHS `tightness` for distribution-shift experiments.

## Branching policies

- **Most fractional:** branch on the variable closest to 0.5.
- **Pseudocost:** maintain historical per-variable down/up unit bound degradations and estimate two-sided branching gains without probing every candidate.
- **Reliability branching:** use pseudocost estimates once both directions have enough observations; otherwise selectively solve the candidate's two child LP relaxations and update its pseudocost statistics.
- **Strong branching:** solve both child LP relaxations for every candidate and choose the strongest bound improvement.
- **MLP learned policy:** rank candidates from inexpensive hand-engineered features.
- **Bipartite GNN policy:** score candidates after variable-to-constraint and constraint-to-variable message passing over the MILP coefficient graph.

The default reliability threshold is `2` observations per direction and can be changed in benchmark scripts with `--reliability-threshold`.

## Imitation data and ranking

Full-tree imitation data follows strong branching through multiple B&B nodes. Candidate features and normalized strong-branching scores are stored per node. Two MLP objectives are available:

- `mse`: regress normalized strong-branching scores.
- `listwise`: cross-entropy over the expert's top-ranked candidate at each node.

Top-1 strong-branching agreement is used as the imitation metric.

## Bipartite GNN representation

The graph model uses variable nodes, constraint nodes, normalized MILP coefficients as bipartite edge weights, normalized variable/constraint features, and two-way message passing before candidate scoring. The GNN is implemented in pure PyTorch. Inference graph construction does **not** call strong branching; expert calls are used only to generate training labels.

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

## Statistical benchmark

```bash
python scripts/benchmark.py \
  --instances-per-seed 20 \
  --vars 12 \
  --constraints 5 \
  --seeds 1000 1001 1002 1003 1004 \
  --reliability-threshold 2 \
  --mse-checkpoint checkpoints/branching_mse.pt \
  --listwise-checkpoint checkpoints/branching_listwise.pt \
  --gnn-checkpoint checkpoints/branching_gnn.pt
```

The comparison includes most-fractional, pseudocost, reliability, strong branching, MLP-MSE, MLP-listwise, and GNN. All policies solve the same generated instances within a seed. Metrics are averaged within seed before cross-seed inference.

For each metric, the benchmark reports mean, sample standard deviation, and a two-sided **Student-t 95% confidence interval** over seed-level replicates. Student-t intervals are used instead of a fixed 1.96 normal approximation because the default number of seeds is small.

The benchmark also reports paired node comparisons:

- comparison-minus-reference node delta,
- percentage node reduction relative to most-fractional,
- percentage node reduction relative to reliability branching where applicable.

Positive `paired_node_reduction_percent` means the comparison policy processed fewer branch-and-bound nodes than the reference policy. Paired summaries preserve the fact that policies are evaluated on the same seed-level instance sets and are generally more informative than comparing two independent marginal confidence intervals.

Wall-clock time remains a secondary metric for these small Python experiments because interpreter, model-inference, and LP-solver overhead can dominate. Node count and LP solve count are the primary algorithmic metrics.

## Generalization benchmark

```bash
python scripts/generalization.py \
  --checkpoint checkpoints/branching_listwise.pt \
  --gnn-checkpoint checkpoints/branching_gnn.pt \
  --reliability-threshold 2 \
  --instances-per-seed 10 \
  --seeds 2000 2001 2002 2003 2004
```

Scenarios include nominal `12v/5c`, larger `16v/5c`, larger `20v/8c`, tighter packing, and looser packing. All active policies solve the same generated instances before repeated-seed summaries are computed.

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

The repository follows the learning-to-branch line associated with Khalil et al. and Gasse et al. The current implementation contains a progression from low-cost hand-designed branching rules through selective strong probing to learned MLP/GNN policies, all evaluated under the same transparent branch-and-bound engine and a paired repeated-seed protocol.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
