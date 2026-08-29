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

## Evaluation

The main solver metrics are:

- branch-and-bound nodes processed,
- LP relaxations solved,
- incumbent objective,
- optimality status,
- branching agreement with strong branching,
- repeated-seed mean/std summaries.

## Research lineage

This repository follows the learning-to-branch line associated with Khalil et al. and Gasse et al., while keeping the first implementation solver-independent and educational.

## License

PolyForm Noncommercial License 1.0.0. Commercial use is not permitted.
