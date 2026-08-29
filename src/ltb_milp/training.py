"""Training utilities for learned branching scores."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ltb_milp.dataset import BranchingDataset


def train_branching_model(
    model: nn.Module,
    dataset: BranchingDataset,
    *,
    epochs: int = 200,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Fit candidate strong-branching scores with mean-squared error."""
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")

    features = torch.as_tensor(dataset.features, dtype=torch.float32)
    targets = torch.as_tensor(dataset.targets, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    model.train()
    for _ in range(epochs):
        predicted = model(features)
        loss = loss_fn(predicted, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.inference_mode()
def predict_scores(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """Predict branching scores for a candidate feature matrix."""
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    return model(tensor).cpu().numpy()
